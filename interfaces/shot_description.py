from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple


class ShotBriefDescription(BaseModel):
    idx: int = Field(
        description="The index of the shot in the sequence, starting from 0.",
        examples=[0, 1, 2],
    )
    is_last: bool = Field(
        description="Whether this is the last shot. If True, the story of the script has ended and no more shots will be planned after this one.",
        examples=[False, True],
    )

    # visual
    cam_idx: int = Field(
        description="The index of the camera in the scene.",
        examples=[0, 1, 2],
    )
    visual_desc: str = Field(
        description='''A concise, filmable shot description using present-tense action, location, framing, and canonical character names. Character identifiers must match the character list and be enclosed in angle brackets (e.g., <Alice>, <Bob>). Name all visible characters rather than substituting appearance-based labels or repeating profiles. Include only brief costume, prop, or physical details essential to this shot.
        Preserve dialogue verbatim and attribute it to the named speaker (e.g., <SLING>: "Gear retracted. Flaps transitioning. Flight path stable. You are clear to climb."). Parentheticals are for essential delivery or action, not biographies.
        ''',
        examples=[
            "An over-the-shoulder shot at eye level, positioned behind <Alice>. The foreground, including <Alice>'s shoulder and head, is softly blurred, directing focus onto <Bob>'s face. <Bob>'s subtle reactions—shifting from surprise to delight—are clearly visible. The supermarket background is gently blurred with cool fluorescent lighting.",
        ]
    )


    # audio
    audio_desc: str = Field(
        description="A detailed description of the audio in the shot.",
        examples=[
            "[Sound Effect] Ambient sound (supermarket background noise, shopping cart wheels rolling)",
            "[Speaker] Alice (Happy): Hello, how are you?",
            None,
        ],
    )

    # sound_effect: Optional[str] = Field(
    #     default=None,
    #     description="The sound effects used in the shot.",
    #     examples=[
    #         "Ambient sound (supermarket background noise, shopping cart wheels rolling)",
    #         None,
    #     ],
    # )
    # speaker: Optional[str] = Field(
    #     default=None,
    #     description="The speaker in the shot, if applicable. If there is no speaker, this field should be set to None.",
    #     examples=[
    #         "Alice",
    #         None,
    #     ],
    # )
    # is_speaker_lip_visible: Optional[bool] = Field(
    #     default=None,
    #     description="Indicates whether the speaker's lips are visible in the shot. If there is no speaker, this field should be set to None.",
    #     examples=[
    #         True,
    #         False,
    #         None,
    #     ],
    # )
    # line: Optional[str] = Field(
    #     default=None,
    #     description="The dialogue or monologue in the shot, if applicable. If there is a speaker, there must be a line. If there is no speaker, this field should be set to None.",
    #     examples=[
    #         "Hello, how are you?",
    #         None,
    #     ],
    # )
    # emotion: Optional[str] = Field(
    #     default=None,
    #     description="The emotion of the speaker when delivering the line, if applicable. If there is a speaker, there must be an emotion. If there is no speaker, this field should be set to None.",
    #     examples=[
    #         "Happy",
    #         None,
    #     ],
    # )

    def __str__(self):
        s = f"Shot {self.idx}:\n"
        s += f"Camera Index: {self.cam_idx}\n"
        s += f"Visual: {self.visual_desc}\n"
        if self.audio_desc:
            s += f"Audio: {self.audio_desc}"
        return s


class ShotDescription(BaseModel):
    idx: int = Field(
        description="The index of the shot in the sequence, starting from 0."
    )
    is_last: bool = Field(
        description="Whether this is the last shot in the sequence. If True, no more shots will be planned after this one."
    )

    # visual
    cam_idx: int = Field(
        description="The index of the camera in the scene.",
        examples=[0, 1, 2],
    )
    visual_desc: str = Field(
        description='''A concise, filmable shot description using present-tense action, location, framing, and canonical character names. Character identifiers must match the character list and be enclosed in angle brackets (e.g., <Alice>, <Bob>). Do not replace names with appearance-based labels or repeat character profiles; keep only brief shot-essential costume, prop, or physical details.
        Preserve dialogue verbatim and attribute it to the named speaker (e.g., <SLING>: "Gear retracted. Flaps transitioning. Flight path stable. You are clear to climb."). Parentheticals are for essential delivery or action, not biographies.''',
        examples=[
            "An over-the-shoulder shot at eye level, positioned behind <Alice>. The foreground, including <Alice>'s shoulder and head, is softly blurred, directing focus onto <Bob>'s face. <Bob>'s subtle reactions—shifting from surprise to delight—are clearly visible. The supermarket background is gently blurred with cool fluorescent lighting.",
        ]
    )
    variation_type: Literal["large", "medium", "small"] = Field(
        description="Indicates the degree of change in the shot's content.",
        examples=["large", "medium", "small"],
    )
    variation_reason: str = Field(
        description="The reason for the variation type of the shot.",
        examples=[
            "This is a transition shot where the content of the first frame and the last frame differs dramatically. So the variation type is large.",
            "Compared to the first frame, a new character appears in the last frame, and there are no significant changes in the composition. So the variation type is medium.",
            "Compared to the first frame, there are only minor changes in the composition. So the variation type is small.",
            "This shot only shows Alice speaking and the changes in her facial expressions, thus the variation type is small.",
        ],
    )

    ff_desc: str = Field(
        description="A static first-frame description using canonical character names, location, framing, composition, and initial poses; no repeated character profiles.",
        examples=[
            "Medium shot at eye level in a supermarket aisle. <Bob> stands on the right in profile, facing right. <Alice> stands on the left behind a shopping cart, looking down. Shelves line both sides under cool fluorescent lighting.",
            "Medium shot at eye level in a meeting room. <Designer A> leans toward <Product Manager B>, who sits with crossed arms. <Development Engineer C> looks between them. <Project Manager D> faces the whiteboard. A glass wall is visible behind them.",
            "Over-the-shoulder shot at eye level in a cafe, behind <Alice>. <Alice>'s shoulder is softly blurred in the foreground; <Bob> faces her across the table. Warm lighting falls across the cafe background.",
        ]
    )
    ff_vis_char_idxs: List[int] = Field(
        default=[],
        description="The indices of the characters in the first frame.",
        examples=[
            [0, 1],
            [0],
            [],
        ],
    )
    set_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit editor binding to known set-reference IDs. None inherits assignments from set descriptors; [] selects no sets and blocks new frame generation. Leave unset during shot planning unless canonical set IDs are already known.",
    )

    reuse_previous_last_frame: Optional[bool] = Field(
        default=None,
        description="When set, explicitly copy the previous active playback shot's last frame as this shot's first frame, or explicitly disable that reuse.",
    )

    lf_desc: str = Field(
        description="A static last-frame description using canonical character names, location, framing, composition, and final poses; no repeated character profiles.",
    )
    lf_vis_char_idxs: List[int] = Field(
        default=[],
        description="The indices of the characters in the last frame.",
    )
    motion_desc: str = Field(
        description='''Concise present-tense camera movement and on-screen action using canonical character names rather than appearance-based labels or repeated profiles. Preserve spoken dialogue verbatim with named speaker cues (e.g., <SLING>: "Gear retracted."). Use short parentheticals only for essential delivery or action. Label narration separately (e.g., Narration: "Everything is looking good.").''',
    )

    # audio
    audio_desc: str = Field(
        description="A detailed description of the audio in the shot.",
        examples=[
            "[Sound Effect] Ambient sound (supermarket background noise, shopping cart wheels rolling)",
            "[Speaker] Alice (Happy): Hello, how are you?",
            None,
        ],
    )
    # sound_effect: Optional[str] = Field(
    #     default=None,
    #     description="The sound effects used in the shot. For example, a door creaking or footsteps approaching.",
    # )
    # speaker: Optional[str] = Field(
    #     default=None,
    #     description="The speaker in the shot, if applicable. If there is no speaker, this field should be set to None.",
    # )
    # is_speaker_lip_visible: Optional[bool] = Field(
    #     default=None,
    #     description="Indicates whether the speaker's lips are visible in the shot. If there is no speaker, this field should be set to None.",
    # )
    # line: Optional[str] = Field(
    #     default=None,
    #     description="The dialogue or monologue in the shot, if applicable. If there is a speaker, there must be a line. If there is no speaker, this field should be set to None.",
    # )
    # emotion: Optional[str] = Field(
    #     default=None,
    #     description="The emotion of the speaker when delivering the line, if applicable. If there is a speaker, there must be an emotion. If there is no speaker, this field should be set to None.",
    # )
