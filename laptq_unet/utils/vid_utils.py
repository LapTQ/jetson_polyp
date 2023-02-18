from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import cv2
import os


class BuilderBase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_product(self):
        pass


class LoaderBase(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]:
        pass

    @abstractmethod
    def get_fps(self) -> float:
        pass

    @abstractmethod
    def get_height(self) -> int:
        pass

    @abstractmethod
    def get_width(self) -> int:
        pass

    @abstractmethod
    def release(self) -> None:
        pass


class VideoLoader(LoaderBase):

    class Builder(BuilderBase):

        def __init__(self, path):
            self.reset()

            cap = cv2.VideoCapture(path)
            self._product.pool = cap
            self._product.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._product.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._product.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._product.fps = cap.get(cv2.CAP_PROP_FPS)


        def reset(self):
            self._product = VideoLoader()

        def get_product(self):
            product = self._product
            self.reset()
            return product

    def __len__(self):
        return self.length

    def get_fps(self) -> float:
        return self.fps

    def get_height(self) -> int:
        return self.height

    def get_width(self) -> int:
        return self.width

    def release(self) -> None:
        self.pool.release()

    def read(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self.pool.read()
        return ret, frame


class ImageLoader(LoaderBase):

    class Builder(BuilderBase):

        def __init__(self, path, is_dir):
            self.reset()

            if is_dir:
                cap = [os.path.join(path, filename) for filename in sorted(os.listdir(path))]
            else:
                cap = [path]
            self._product.pool = cap
            H, W = cv2.imread(cap[0]).shape[:2]
            self._product.height = H
            self._product.width = W
            self._product.length = len(cap)
            self._product.fps = 1.
            self._product.current_frame = 0

        def reset(self):
            self._product = ImageLoader()

        def get_product(self):
            product = self._product
            self.reset()
            return product

    def __len__(self):
        return self.length

    def get_fps(self) -> float:
        return self.fps

    def get_height(self) -> int:
        return self.height

    def get_width(self) -> int:
        return self.width

    def release(self) -> None:
        pass

    def read(self) -> Tuple[bool, np.ndarray]:
        if self.current_frame < len(self.pool):
            frame = cv2.imread(self.pool[self.current_frame])
            self.current_frame += 1
            ret = True
        else:
            frame = None
            ret = False
        return ret, frame

