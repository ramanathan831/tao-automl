# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prompt templates for LLM-powered AutoML capabilities."""

from tao_automl.brain.prompts.llm_brain_prompts import (
    build_recommendation_prompt,
    build_recommendation_with_reasoning_prompt,
)
from tao_automl.brain.prompts.analyzer_prompts import (
    build_analysis_prompt,
)
from tao_automl.brain.prompts.nl_config_prompts import (
    build_nl_config_prompt,
)
from tao_automl.brain.prompts.autoresearch_prompts import (
    build_autoresearch_prompt,
    build_keep_discard_prompt,
    build_hybrid_strategy_prompt,
    build_prescreen_prompt,
    build_spec_verification_prompt,
    build_result_verification_prompt,
    build_knowledge_summary_prompt,
)

__all__ = [
    "build_recommendation_prompt",
    "build_recommendation_with_reasoning_prompt",
    "build_analysis_prompt",
    "build_nl_config_prompt",
    "build_autoresearch_prompt",
    "build_keep_discard_prompt",
    "build_hybrid_strategy_prompt",
    "build_prescreen_prompt",
    "build_spec_verification_prompt",
    "build_result_verification_prompt",
    "build_knowledge_summary_prompt",
]
