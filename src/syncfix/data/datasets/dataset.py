from typing import Callable, List, Union, Dict, Tuple

import hashlib
import random
from dataclasses import dataclass

import pytorch_lightning as pl
import webdataset as wds
from webdataset import DataPipeline

from ..filters import BaseFilter, FilterWrapper
from ..mappers import BaseMapper, MapperWrapper
from .collation_fn import custom_collation_fn
from .datasets_config import DataModuleConfig

@dataclass
class TrainMultiViewConfig:
    """Configuration for pairing N temporally-nearby frames into a single sample during training.

    The __key__ of each sample is expected to be of the form:
        "{scene}__{K_dir}__{run_dir}__{it_tag}__{frame_index}"

    We define a sequence id as the first four fields, and frame_index as the last.
    """
    enabled: bool = False
    window: int = 3
    seed: int = 0
    num_views: int = 2  # number of views stacked into one sample

    # input/target keys present in the *single-view* sample dict produced by tarfile_to_samples
    input_key: str = "png"
    target_key: str = "gt.png"
    ref_key: str = "ref.png"

    # Output keys are generated dynamically (img1.png, img2.png ... imgN.png)


class TrainMultiViewStage:
    """Stateful WebDataset mapper that pairs each sample with N-1 nearby frames.

    Important design choice for distributed stability:
      - For every incoming sample, we ALWAYS emit exactly one paired sample.
        If not enough neighbors exist yet, we pad by pairing the sample with itself.
      - This avoids changing the number of yielded samples due to buffering.
    """

    def __init__(self, cfg: TrainMultiViewConfig):
        self.cfg = cfg
        # seq_id -> {frame_idx: (inp, tgt, ref)}
        self._cache: Dict[str, Dict[int, Tuple]] = {}

    @staticmethod
    def _parse_key(sample_key: str) -> Tuple[str, int]:
        parts = sample_key.split("__")
        if len(parts) < 5:
            raise ValueError(f"Expected __key__ with 5+ fields split by '__', got: {sample_key}")
        seq_id = "__".join(parts[:-1])
        frame_idx = int(parts[-1].split('_')[-1])
        return seq_id, frame_idx

    def _rng_for(self, sample_key: str) -> random.Random:
        # Stable per-sample RNG: seed + hash(sample_key)
        h = hashlib.blake2b((str(self.cfg.seed) + "|" + sample_key).encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(h, "little")
        return random.Random(v)

    def __call__(self, sample: Dict) -> Dict:
        if not self.cfg.enabled:
            return sample

        key = sample.get("__key__")
        if key is None:
            raise ValueError("WebDataset sample missing '__key__'")

        seq_id, idx = self._parse_key(key)

        # Resolve input/target keys
        if self.cfg.input_key not in sample or self.cfg.target_key not in sample:
            raise KeyError(f"Required keys not found in sample keys: {list(sample.keys())}")

        inp_bytes = sample[self.cfg.input_key]
        tgt_bytes = sample[self.cfg.target_key]
        ref_bytes = sample[self.cfg.ref_key]

        # 1. Update LRU Cache
        cache = self._cache.pop(seq_id, {})
        self._cache[seq_id] = cache

        if len(self._cache) > 10:
            oldest_seq_id = next(iter(self._cache))
            del self._cache[oldest_seq_id]

        cache[idx] = (inp_bytes, tgt_bytes, ref_bytes)

        # 2. Prune current sequence cache to keep memory bounded
        if self.cfg.window > 0:
            lo, hi = idx - self.cfg.window, idx + self.cfg.window
            if len(cache) > 10: 
                for k in list(cache.keys()):
                    if k < lo or k > hi:
                        cache.pop(k, None)

        # 3. Find candidates & sample N-1 neighbors
        rng = self._rng_for(key)
        candidates = [j for j in cache.keys() if j != idx and abs(j - idx) <= self.cfg.window]
        sampled_neighbors = []
        if self.cfg.num_views > 1:
            k_needed = self.cfg.num_views - 1
            
            if len(candidates) >= k_needed:
                # 1. Enough candidates: sample WITHOUT replacement
                sampled_neighbors = rng.sample(candidates, k=k_needed)
                
            elif len(candidates) > 0:
                # 2. Some candidates, but not enough: use all existing, then sample to fill the gap
                sampled_neighbors = list(candidates)
                remainder = k_needed - len(candidates)
                sampled_neighbors.extend(rng.choices(candidates, k=remainder))
                
            else:
                # 3. No candidates at all (e.g., first frame): pad with itself
                sampled_neighbors = [idx] * k_needed

        # 4. Combine current frame + neighbors
        selected_idxs = [idx] + sampled_neighbors

        # 5. Randomly choose ONE frame to provide the single Reference Image 
        ref_provider_idx = rng.choice(selected_idxs)
        ref_provider_data = cache[ref_provider_idx]

        # 6. Build the output sample dynamically
        out: Dict = {
            "__key__": key, 
            "__url__": sample.get("__url__")
        }

        # Set the unified reference
        out[self.cfg.ref_key] = ref_provider_data[2]

        # Loop through the selected views to build img1, img2, etc. (1-based index)
        for i, s_idx in enumerate(selected_idxs):
            view_id = i + 1  # 1, 2, 3...
            s_data = cache[s_idx]

            out[f"img{view_id}.png"] = s_data[0]
            out[f"gt{view_id}.png"] = s_data[1]

        return out



class DataPipeline:
    """
    DataPipeline class for creating a dataloader from a single configuration

    Args:

        config (DataModuleConfig):
            Configuration for the dataset

        filters_mappers (Union[List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the dataset. These will be sequentially applied.

        batched_filters_mappers (List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of batched transforms for the dataset. These will be sequentially applied.
    """

    def __init__(
        self,
        config: DataModuleConfig,
        filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ],
        batched_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
    ):
        self.config = config
        self.shards_path_or_urls = config.shards_path_or_urls
        self.filters_mappers = filters_mappers
        self.batched_filters_mappers = batched_filters_mappers or []

        if filters_mappers is None:
            filters_mappers = []

        # set processing pipeline
        self.processing_pipeline = [wds.decode(config.decoder, handler=config.handler)]
        self.processing_pipeline.extend(
            self._add_filters_mappers(
                filters_mappers=filters_mappers,
                handler=config.handler,
            )
        )

    def _add_filters_mappers(
        self,
        filters_mappers: List[
            Union[
                FilterWrapper,
                MapperWrapper,
            ]
        ],
        handler: Callable = wds.warn_and_continue,
    ) -> List[Union[FilterWrapper, MapperWrapper]]:
        tmp_pipeline = []
        for filter_mapper in filters_mappers:
            if isinstance(filter_mapper, FilterWrapper) or isinstance(
                filter_mapper, BaseFilter
            ):
                tmp_pipeline.append(wds.select(filter_mapper))
            elif isinstance(filter_mapper, MapperWrapper) or isinstance(
                filter_mapper, BaseMapper
            ):
                tmp_pipeline.append(wds.map(filter_mapper, handler=handler))
            elif isinstance(filter_mapper) or isinstance(filter_mapper):
                tmp_pipeline.append(wds.map(filter_mapper, handler=handler))
            else:
                raise ValueError("Unknown type of filter/mapper")
        return tmp_pipeline

    def setup(self):
        pipeline = [wds.SimpleShardList(self.shards_path_or_urls)]
        # pipeline = [wds.ResampledShards(self.shards_path_or_urls)]
        # shuffle before split by node
        if self.config.shuffle_before_split_by_node_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_before_split_by_node_buffer_size,
                    handler=self.config.handler,
                )
            )
        # split by node
        pipeline.append(wds.split_by_node)

        # shuffle before split by workers
        if self.config.shuffle_before_split_by_workers_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_before_split_by_workers_buffer_size,
                    handler=self.config.handler,
                )
            )
        # split by worker
        pipeline.extend(
            [
                wds.split_by_worker,
                wds.tarfile_to_samples(
                    handler=self.config.handler,
                    rename_files=self.config.rename_files_fn,
                ),
            ]
        )

        # multi-view pairing stage
        pair_num_views = int(getattr(self.config, "pair_num_views", 1))
        if pair_num_views > 1:
            pair_cfg = TrainMultiViewConfig(
                enabled=True,
                window=int(getattr(self.config, "pair_window", 3)),
                seed=int(getattr(self.config, "pair_seed", 0)),
                num_views=pair_num_views,
            )
            pipeline.append(wds.map(TrainMultiViewStage(pair_cfg), handler=self.config.handler))


        # shuffle before filter mappers
        if self.config.shuffle_before_filter_mappers_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_before_filter_mappers_buffer_size,
                    handler=self.config.handler,
                )
            )

        # apply filters and mappers
        pipeline.extend(self.processing_pipeline)

        # shuffle after filter mappers
        if self.config.shuffle_after_filter_mappers_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_after_filter_mappers_buffer_size,
                    handler=self.config.handler,
                ),
            )

        # batching
        pipeline.append(
            wds.batched(
                self.config.per_worker_batch_size,
                partial=False,
                collation_fn=custom_collation_fn,
            )
        )

        # apply batched transforms
        pipeline.extend(
            self._add_filters_mappers(
                filters_mappers=self.batched_filters_mappers,
                handler=self.config.handler,
            )
        )

        # create the data pipeline
        pipeline = wds.DataPipeline(*pipeline, handler=self.config.handler) #.repeat()
        
        #.with_epoch(10000)

        # set the pipeline
        self.pipeline = pipeline

    def dataloader(self):
        # return the loader
        return wds.WebLoader(
            self.pipeline,
            batch_size=None,
            num_workers=self.config.num_workers,
        ) #.with_epoch(1000)


class DataModule(pl.LightningDataModule):
    """
    Main DataModule class for creating data loaders and training/evaluating models

    Args:

        train_config (DataModuleConfig):
            Configuration for the training dataset

        train_filters_mappers (Union[List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the training dataset. These will be sequentially applied.

        train_batched_filters_mappers (List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of batched transforms for the training dataset. These will be sequentially applied.

        eval_config (DataModuleConfig):
            Configuration for the evaluation dataset

        eval_filters_mappers (List[Union[FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the evaluation dataset.These will be sequentially applied.

        eval_batched_filters_mappers (List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of batched transforms for the evaluation dataset. These will be sequentially applied.
    """

    def __init__(
        self,
        train_config: DataModuleConfig,
        train_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
        train_batched_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
        eval_config: DataModuleConfig = None,
        eval_filters_mappers: List[Union[FilterWrapper, MapperWrapper]] = None,
        eval_batched_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
    ):
        super().__init__()

        self.train_config = train_config
        self.train_filters_mappers = train_filters_mappers
        self.train_batched_filters_mappers = train_batched_filters_mappers

        self.eval_config = eval_config
        self.eval_filters_mappers = eval_filters_mappers
        self.eval_batched_filters_mappers = eval_batched_filters_mappers

    def setup(self, stage=None):
        """
        Setup the data module and create the webdataset processing pipelines
        """

        # train pipeline
        self.train_pipeline = DataPipeline(
            config=self.train_config,
            filters_mappers=self.train_filters_mappers,
            batched_filters_mappers=self.train_batched_filters_mappers,
        )
        self.train_pipeline.setup()
        # eval pipeline
        if self.eval_config is not None:
            self.eval_pipeline = DataPipeline(
                config=self.eval_config,
                filters_mappers=self.eval_filters_mappers,
                batched_filters_mappers=self.eval_batched_filters_mappers,
            )
            self.eval_pipeline.setup()

    def train_dataloader(self):
        return self.train_pipeline.dataloader()

    def val_dataloader(self):
        return self.eval_pipeline.dataloader()
