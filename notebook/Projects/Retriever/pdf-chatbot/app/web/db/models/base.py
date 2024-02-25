from abc import abstractmethod
from typing import Any, List, Optional, Type, TypeVar

from app.web.db import db

T = TypeVar("T", bound="BaseModel")


class BaseModel(db.Model):
    __abstract__ = True

    @classmethod
    def create(cls: Type[T], commit=True, **kwargs) -> T:
        instance = cls(**kwargs)
        return instance.save(commit)

    @classmethod
    def find_by(cls: Type[T], **kwargs) -> Optional[T]:
        return db.session.execute(db.select(cls).filter_by(**kwargs)).scalar_one()

    @classmethod
    def where(cls: Type[T], **kwargs) -> List[T]:
        return db.session.execute(db.select(cls).filter_by(**kwargs)).scalars().all()

    @classmethod
    def upsert(cls: Type[T], commit=True, **kwargs) -> T:
        instance = None
        if kwargs.get("id"):
            instance = cls.find_by(id=kwargs["id"])

        if instance:
            instance.update(commit, **kwargs)
            return instance
        instance = cls.create(**kwargs)
        return instance

    @classmethod
    def delete_by(cls, commit: bool = True, **kwargs) -> None:
        instance = cls.find_by(**kwargs)
        db.session.delete(instance)
        if commit:
            return db.session.commit()
        return

    @classmethod
    def as_dicts(cls, models) -> list[dict[str, Any]]:
        return [m.as_dict() for m in models]

    @abstractmethod
    def as_dict(self) -> None:
        raise NotImplementedError

    def update(self, commit=True, **kwargs) -> None:
        for attr, value in kwargs.items():
            if attr != ["id"]:
                setattr(self, attr, value)
        if commit:
            return self.save()
        return self

    def save(self, commit=True) -> T:
        db.session.add(self)
        if commit:
            db.session.commit()
        return self
