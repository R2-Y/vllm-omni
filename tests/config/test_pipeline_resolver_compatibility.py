# SPDX-License-Identifier: Apache-2.0

from transformers import PretrainedConfig

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import PipelineConfig, StagePipelineConfig


def test_all_registry_resolvers_keep_key_only_topology_compatibility():
    resolver_keys = [key for key, value in OMNI_PIPELINES.items() if callable(value)]

    for key in resolver_keys:
        pipeline = resolve_pipeline_config(key)
        assert pipeline is not None, key
        assert pipeline.stages, key


def test_all_registry_resolvers_keep_wrong_config_topology_compatibility():
    unrelated = PretrainedConfig()
    resolver_keys = [key for key, value in OMNI_PIPELINES.items() if callable(value)]

    for key in resolver_keys:
        pipeline = resolve_pipeline_config(key, unrelated)
        assert pipeline is not None, key
        assert pipeline.stages, key


def test_architecture_fallback_does_not_execute_unrelated_resolver(monkeypatch):
    unrelated_topology = PipelineConfig(
        model_type="unrelated",
        hf_architectures=("OtherArchitecture",),
        stages=(StagePipelineConfig(stage_id=0, model_stage="unrelated"),),
    )

    def unrelated_resolver(_config):
        raise AssertionError("unrelated resolver must not run")

    unrelated_resolver.default_pipeline_config = unrelated_topology
    target = PipelineConfig(
        model_type="target",
        hf_architectures=("TargetArchitecture",),
        stages=(StagePipelineConfig(stage_id=0, model_stage="target"),),
    )
    monkeypatch.setattr(
        "vllm_omni.config.config_factory.OMNI_PIPELINES",
        {
            "unrelated": unrelated_resolver,
            "target": target,
        },
    )
    config = PretrainedConfig(architectures=["TargetArchitecture"])
    monkeypatch.setattr(StageConfigFactory, "try_infer_model_type", classmethod(lambda cls, **kwargs: "unknown"))
    monkeypatch.setattr(StageConfigFactory, "get_hf_config", classmethod(lambda cls, **kwargs: config))

    assert StageConfigFactory.get_pipeline_config("model", False) is target
