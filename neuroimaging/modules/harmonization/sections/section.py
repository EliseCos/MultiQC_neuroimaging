from dataclasses import dataclass
from abc import ABCMeta, abstractmethod


@dataclass
class HtmlContent:
    content: str
    metadata: dict


class Section(metaclass=ABCMeta):
    def __init__(self):
        pass

    @property
    def name(self):
        return None

    @property
    def anchor(self):
        return None

    @property
    def description(self):
        return ""

    @abstractmethod
    def build_html(self) -> HtmlContent:
        raise NotImplementedError()


class SectionBundleMetric(Section):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_metrics(self):
        raise NotImplementedError()

    @abstractmethod
    def filter_metrics(self, metrics):
        raise NotImplementedError()

    @abstractmethod
    def get_bundles(self):
        raise NotImplementedError()

    @abstractmethod
    def filter_bundles(self, bundles):
        raise NotImplementedError()
