import logging
import os
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from interfaces import CharacterInScene, ImageOutput
from langchain_core.messages import HumanMessage, SystemMessage



prompt_template_front = \
"""
Generate a full-body, front-view portrait of character {identifier} based on the following description, with a pure white background. Use a wide 16:9 landscape canvas, not a vertical portrait canvas. The character should be centered in the image, occupying the middle of the wide frame with enough horizontal empty space. Gazing straight ahead. Standing with arms relaxed at sides. Natural expression.
Features: {features}
Style: {style}
"""

prompt_template_side = \
"""
Generate a full-body, side-view portrait of character {identifier} based on the provided front-view portrait, with a pure white background. Use a wide 16:9 landscape canvas, not a vertical portrait canvas. The character should be centered in the image, occupying the middle of the wide frame with enough horizontal empty space. Facing left. Standing with arms relaxed at sides.
"""

prompt_template_back = \
"""
Generate a full-body, back-view portrait of character {identifier} based on the provided front-view portrait, with a pure white background. Use a wide 16:9 landscape canvas, not a vertical portrait canvas. The character should be centered in the image, occupying the middle of the wide frame with enough horizontal empty space. No facial features should be visible.
"""


class CharacterPortraitsGenerator:
    def __init__(
        self,
        image_generator,
    ):
        self.image_generator = image_generator
        # The prompt behind each portrait, keyed by (character, view). Portraits are
        # generated on a plain white backdrop with no other prompt artifact, so without
        # this there is nothing to inspect when one of them comes out wrong.
        self.prompts: Dict[tuple, str] = {}

    async def _generate(self, description: str, **kwargs) -> ImageOutput:
        """Generate one portrait, naming the character and view on failure.

        Provider-side rejections do not appear anywhere in the call arguments, so
        without this the failing portrait is only implied by the last progress
        event emitted by the calling pipeline.
        """
        try:
            return await self.image_generator.generate_single_image(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"Image generation failed for {description}: {exc}") from exc

    async def generate_front_portrait(
        self,
        character: CharacterInScene,
        style: str,
    ) -> ImageOutput:
        features = "(static) " + (character.static_features or "") + "; (dynamic) " + (character.dynamic_features or "")
        prompt = prompt_template_front.format(
            identifier=character.identifier_in_scene,
            features=features,
            style=style,
        )
        self.prompts[(character.identifier_in_scene, "front")] = prompt
        return await self._generate(
            f"the front portrait of {character.identifier_in_scene}",
            prompt=prompt,
            # size="512x512",
        )

    async def generate_side_portrait(
        self,
        character: CharacterInScene,
        front_image_path: str,
    ) -> ImageOutput:
        prompt = prompt_template_side.format(
            identifier=character.identifier_in_scene,
        )
        self.prompts[(character.identifier_in_scene, "side")] = prompt
        return await self._generate(
            f"the side portrait of {character.identifier_in_scene}",
            prompt=prompt,
            reference_image_paths=[front_image_path],
            # size="1024x1024",
        )


    async def generate_back_portrait(
        self,
        character: CharacterInScene,
        front_image_path: str,
    ) -> ImageOutput:
        prompt = prompt_template_back.format(
            identifier=character.identifier_in_scene,
        )
        self.prompts[(character.identifier_in_scene, "back")] = prompt
        return await self._generate(
            f"the back portrait of {character.identifier_in_scene}",
            prompt=prompt,
            reference_image_paths=[front_image_path],
            # size="512x512",
        )