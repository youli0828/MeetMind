from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Segment(Model):
    id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1)
    speaker: str | None

    @model_validator(mode="after")
    def valid_interval(self):
        if self.end < self.start:
            raise ValueError("segment end must be >= start")
        if not self.text.strip():
            raise ValueError("segment text must not be blank")
        return self


class Transcript(Model):
    language: str | None
    segments: list[Segment] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_segments(self):
        ids = [s.id for s in self.segments]
        starts = [s.start for s in self.segments]
        if len(set(ids)) != len(ids):
            raise ValueError("segment IDs must be unique")
        if starts != sorted(starts):
            raise ValueError("segments must be ordered by start time")
        return self


class Topic(Model):
    title: str = Field(min_length=1)
    discussion: str = Field(min_length=1)
    source_segment_ids: list[int] = Field(min_length=1)


class Decision(Model):
    text: str = Field(min_length=1)
    source_segment_ids: list[int] = Field(min_length=1)


class ActionItem(Model):
    task: str = Field(min_length=1)
    owner: str | None
    due: str | None
    source_segment_ids: list[int] = Field(min_length=1)


class Minutes(Model):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    topics: list[Topic]
    decisions: list[Decision]
    action_items: list[ActionItem]

    def validate_sources(self, transcript: Transcript) -> None:
        valid_ids = {s.id for s in transcript.segments}
        for item in [*self.topics, *self.decisions, *self.action_items]:
            if not set(item.source_segment_ids) <= valid_ids:
                raise ValueError("minutes reference unknown segment IDs")
