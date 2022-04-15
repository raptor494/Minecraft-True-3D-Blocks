#!python
from __future__ import annotations
from abc import abstractmethod
from enum import Enum, auto
import typing; from typing import Any, Callable, Generator, Iterable, Literal, NamedTuple, TypeAlias, Sequence, TypeVar, Generic, cast, overload
import dataclasses; from dataclasses import dataclass
from math import gcd
import itertools
import functools
import re
import argparse
import sys; from sys import exit
import textwrap
from pathlib import Path
from json_utils import json
from PIL import Image as image
from PIL.Image import Image

N = TypeVar('N', int, float)
T = TypeVar('T')

@overload
def iround(number: int, ndigits: int | None=...) -> int: ...
@overload
def iround(number: float, ndigits: None=...) -> int: ...
@overload
def iround(number: float, ndigits: int) -> int | float: ...

def iround(number: int | float, ndigits: int=None):
    return makeinteger(round(number, ndigits))

def makeinteger(val: int | float) -> int | float:
    if isinstance(val, float) and val.is_integer():
        return int(val)
    else:
        assert isinstance(val, int), type(val)
        return val

@dataclass
class Rectangle(Generic[T]):
    x: int # top-left x coordinate
    y: int # top-left y coordinate
    width: int # width to the right
    height: int # height down
    data: T

# class FaceType(Enum):
#     NONE = auto()
#     SOLID = auto()
#     CULL = auto()

ColorRGBA = tuple[int, int, int, int]

class FaceType(Enum):
    SOLID = auto()
    OVERLAY = auto()
    GLASS = auto()
    OVERLAY_AND_GLASS = auto()

class Mirror(Enum):
    HORIZONTAL = auto()

@dataclass
class FaceData:
    depth: int | float # depth of the face
    type: FaceType = FaceType.SOLID
    depth2: int | float | None = None # depth of the second face, if any
    mirror: Mirror | None = None

    def __setattr__(self, name: str, value):
        match name:
            case 'depth':
                if not isinstance(value, int | float):
                    raise TypeError("depth must be int or float")
            case 'type':
                if not isinstance(value, FaceType):
                    raise TypeError(f"type must be a FaceType")
            case 'depth2':
                if not isinstance(value, int | float | None):
                    raise TypeError("depth2 must be int, float, or None")
            case 'mirror':
                if not isinstance(value, Mirror | None):
                    raise TypeError("mirror must be a Mirror or None")
        super().__setattr__(name, value)
    
def face_data_from_color(color: ColorRGBA) -> FaceData:
    if color[3] == 0:
        raise ValueError("transparent color would result in no face")
    if color == (0, 0, 0, 255):
        return FaceData(depth=3)
    red, green, blue, alpha = color
    assert all(isinstance(c, int) for c in color), color
    assert 0 <= red <= 255, red
    assert 0 <= green <= 255, green
    assert 0 <= blue <= 255, blue
    assert 0 <= alpha <= 255, alpha
    d2 = None
    match red:
        case 0:
            d = 3
        case 64:
            d = 2
        case 128:
            d = 1
        case 255:
            d = 0
        case _:
            raise ValueError(f"invalid red value {red:d}, must be either 0, 64, 128, or 255")
    if green == red:
        m = None
    elif green == 255 - red or green == 256 - red:
        m = Mirror.HORIZONTAL
    else:
        raise ValueError(f"invalid green value {green:d}, must be either {red:d} or {255 - red:d}")
    match alpha:
        case 255:
            f = FaceType.SOLID
        case 192:
            f = FaceType.OVERLAY_AND_GLASS
            match blue:
                case 0:
                    d2 = 3
                case 64:
                    d2 = 2
                case 128:
                    d2 = 1
                case 255:
                    d2 = 0
                case _:
                    raise ValueError(f"invalid blue value {blue:d}, must be either 0, 64, 128, or 255 when alpha value is 192")
        case 128:
            f = FaceType.OVERLAY
        case 64:
            f = FaceType.GLASS
        case _:
            raise ValueError(f"invalid alpha value {alpha:d}")

    return FaceData(depth=d, type=f, depth2=d2, mirror=m)
        
FindRectangleDirectionVertical = Literal["top-bottom", "bottom-top"]
FindRectangleDirectionHorizontal = Literal["left-right", "right-left"]
FindRectangleDirection = tuple[FindRectangleDirectionVertical, FindRectangleDirectionHorizontal] | tuple[FindRectangleDirectionHorizontal, FindRectangleDirectionVertical]

def getcolor(color: int | tuple[int, int, int, int]) -> ColorRGBA:
    match color:
        case int():
            return ((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF, (color >> 24) & 0xFF)
        case (int(r), int(g), int(b), int(a)):
            assert 0 <= r <= 255, f"red out of bounds: {r}"
            assert 0 <= g <= 255, f"green out of bounds: {g}"
            assert 0 <= b <= 255, f"blue out of bounds: {b}"
            assert 0 <= b <= 255, f"alpha out of bounds: {a}"
            return color
        case _:
            raise TypeError

def find_rectangle(img: Image, x: int, y: int, grid: Sequence[Sequence[Rectangle[ColorRGBA] | None]], direction: FindRectangleDirection=("left-right", "top-bottom")) -> Rectangle[ColorRGBA]:
    assert x < img.width and y < img.height, f"(x, y) = ({x:d}, {y:d}), (img.width, img.height) = ({img.width:d}, {img.height:d})"
    color: ColorRGBA = getcolor(img.getpixel((x, y)))
    if color[3] == 0:
        color = (0, 0, 0, 0)
        def samecolor(clr):
            return getcolor(clr)[3] == 0
    else:
        def samecolor(clr):
            return getcolor(clr) == color

    width = height = 0

    match direction[0]:
        case "left-right":
            width = 1
            while x + width < img.width and samecolor(img.getpixel((x + width, y))) and grid[y][x + width] is None:
                width += 1
            if direction[1] not in typing.get_args(FindRectangleDirectionVertical):
                raise ValueError("invalid direction")
        case "right-left":
            width = 1
            while x - width >= 0 and samecolor(img.getpixel((x - width, y))) and grid[y][x - width] is None:
                width += 1
            x -= width - 1
            if direction[1] not in typing.get_args(FindRectangleDirectionVertical):
                raise ValueError("invalid direction")
        case "top-bottom":
            height = 1
            while y + height < img.height and samecolor(img.getpixel((x, y + height))) and grid[y + height][x] is None:
                height += 1
            if direction[1] not in typing.get_args(FindRectangleDirectionHorizontal):
                raise ValueError("invalid direction")
        case "bottom-top":
            height = 1
            while y - height >= 0 and samecolor(img.getpixel((x, y - height))) and grid[y - height][x] is None:
                height += 1
            y -= height - 1
            if direction[1] not in typing.get_args(FindRectangleDirectionHorizontal):
                raise ValueError("invalid direction")
        case _:
            raise ValueError("invalid direction")

    match direction[1]:
        case "top-bottom":
            height = 1
            while y + height < img.height:
                valid = True
                for w in range(width):
                    if not samecolor(img.getpixel((x + w, y + height))) or grid[y + height][x + w] is not None:
                        valid = False
                        break
                if valid:
                    height += 1
                else:
                    break
        case "bottom-top":
            height = 1
            while y - height >= 0:
                valid = True
                for w in range(width):
                    if not samecolor(img.getpixel((x + w, y - height))) or grid[y - height][x + w] is not None:
                        valid = False
                        break
                if valid:
                    height += 1
                else:
                    break
            y -= height - 1
        case "left-right":
            width = 1
            while x + width < img.width:
                valid = True
                for h in range(height):
                    if not samecolor(img.getpixel((x + width, y + h))) or grid[y + h][x + width] is not None:
                        valid = False
                        break
                if valid:
                    width += 1
                else:
                    break
        case "right-left":
            width = 1
            while x - width >= 0:
                valid = True
                for h in range(height):
                    if not samecolor(img.getpixel((x - width, y + h))) or grid[y + h][x - width] is not None:
                        valid = False
                        break
                if valid:
                    width += 1
                else:
                    break
            x -= width - 1
        case _:
            raise ValueError("invalid direction")
            
    assert 0 <= x < img.width, f"{x = }, {img.width = }"
    assert 0 <= y < img.height, f"{y = }, {img.height = }"
    assert width > 0, width
    assert height > 0, height

    return Rectangle(x, y, width, height, color)

def get_rectangles(img: Image, direction: FindRectangleDirection=("left-right", "top-bottom"), print_grid: bool=False) -> list[Rectangle[ColorRGBA]]:
    rectangles: list[Rectangle[ColorRGBA]] = []

    initial_x = initial_y = 0
    x = y = 0
    dx = dy = 0

    for d in direction:
        match d:
            case "left-right":
                initial_x = 0
                dx = 1
            case "right-left":
                initial_x = img.width - 1
                dx = -1
            case "top-bottom":
                initial_y = 0
                dy = 1
            case "bottom-top":
                initial_y = img.height - 1
                dy = -1
            case _:
                raise ValueError("invalid direction")
    
    if dx == 0 or dy == 0:
        raise ValueError("invalid direction")

    rectangle_grid: list[list[Rectangle[ColorRGBA] | None]] = [[None]*img.width for _ in range(img.height)]

    def add_rect(r: Rectangle[ColorRGBA], /):
        for y in range(r.y, r.y + r.height):
            for x in range(r.x, r.x + r.width):
                assert rectangle_grid[y][x] is None
                rectangle_grid[y][x] = r
        rectangles.append(r)

    match direction[0]:
        case "left-right" | "right-left":
            for y_add in range(img.height):
                y = initial_y + dy * y_add
                for x_add in range(img.width):
                    x = initial_x + dx * x_add
                    if rectangle_grid[y][x] is None:
                        r = find_rectangle(img, x, y, rectangle_grid, direction)
                        add_rect(r)
        case "top-bottom" | "bottom-top":
            for x_add in range(img.width):
                x = initial_x + dx * x_add
                for y_add in range(img.height):
                    y = initial_y + dy * y_add
                    if rectangle_grid[y][x] is None:
                        r = find_rectangle(img, x, y, rectangle_grid, direction)
                        add_rect(r)

    if print_grid:
        from collections import defaultdict
        counter = 1
        def next_counter():
            nonlocal counter
            counter += 1
            return counter
        ids = defaultdict(next_counter)
        print("  ", end="")
        for i in range(len(rectangle_grid[0])):
            print(f"|{i:02d}", end="")
        print()
        for i, row in enumerate(rectangle_grid):
            print(f"{i:02d}|", end="")
            for col in row:
                if col is None or col.data[3] == 0:
                    print("\33[0m   ", end="")
                else:
                    print(f"\33[48;5;{ids[id(col)]}m   ", end="")
            print("\33[0m")
        print("\33[0m-")

    return rectangles

def get_faces(rectangles: Iterable[Rectangle[ColorRGBA]]) -> list[Rectangle[FaceData]]:
    faces: list[Rectangle[FaceData]] = []
    for r in rectangles:
        if r.data[3]:
            faces.append(Rectangle(r.x, r.y, r.width, r.height, face_data_from_color(r.data)))
    return faces

def get_grid(rectangles: Iterable[Rectangle[T]], width: int, height: int) -> tuple[tuple[Rectangle[T] | None, ...], ...]:
    grid: list[list[Rectangle[T] | None]] = [[None]*width for _ in range(height)]
    for r in rectangles:
        for y in range(r.y, r.y + r.height):
            for x in range(r.x, r.x + r.width):
                try:
                    if grid[y][x] is not None:
                        raise ValueError(f"some rectangles overlap: {r} and {grid[y][x]}")
                except IndexError:
                    raise IndexError(f"{y = }, {x = }, {len(grid) = }, {len(grid[0]) = }")
                grid[y][x] = r
    return tuple(map(tuple, grid))

class Direction(Enum):
    NORTH = auto()
    SOUTH = auto()
    EAST = auto()
    WEST = auto()
    UP = auto()
    DOWN = auto()

@dataclass
class UV:
    x1: int | float
    y1: int | float
    x2: int | float
    y2: int | float

    def clone(self):
        return UV(self.x1, self.y1, self.x2, self.y2)

    def __setattr__(self, name: str, value):
        if name in ('x1', 'y1', 'x2', 'y2'):
            value = iround(value, ndigits=3)
        super().__setattr__(name, value)

TEXTURE_REGEX = re.compile(r'#[a-z_0-9]+')

@dataclass
class Face:
    texture: str
    uv: UV | None = None
    cullface: Direction | None = None
    rotation: int = 0 # 0, 90, 180, 270

    def __init__(self, texture: str, uv: UV | tuple[int | float, int | float, int | float, int | float] | list[int | float] | None=None, cullface: Direction=None, rotation: int=0):
        self.texture = texture
        match uv:
                case UV() | None:
                    pass
                case [int(x1) | float(x1), int(y1) | float(y1), int(x2) | float(x2), int(y2) | float(y2)]:
                    x1 = makeinteger(x1)
                    y1 = makeinteger(y1)
                    x2 = makeinteger(x2)
                    y2 = makeinteger(y2)
                    uv = UV(x1, y1, x2, y2)
                case _:
                    raise TypeError("invalid uv value, must be UV object or sequence of 4 ints or floats")
        self.uv = uv
        self.cullface = cullface
        self.rotation = rotation

    def clone(self):
        return Face(self.texture, self.uv.clone() if self.uv else None, self.cullface)

    def __setattr__(self, name: str, value):
        match name:
            case 'texture':
                if not isinstance(value, str):
                    raise TypeError("texture must be a string")
                if not TEXTURE_REGEX.fullmatch(value):
                    raise ValueError("texture must start with a # and contain only lowercase letters and numbers")
            case 'uv':
                match value:
                    case UV() | None:
                        pass
                    case [int(x1) | float(x1), int(y1) | float(y1), int(x2) | float(x2), int(y2) | float(y2)]:
                        x1 = makeinteger(x1)
                        y1 = makeinteger(y1)
                        x2 = makeinteger(x2)
                        y2 = makeinteger(y2)
                        value = UV(x1, y1, x2, y2)
                    case _:
                        raise TypeError("invalid uv value, must be UV object or sequence of 4 ints or floats")
            case 'cullface':
                if not isinstance(value, Direction | None):
                    raise TypeError("cullface must be a Direction or None")
            case 'rotation':
                if value is None:
                    value = 0
                else:
                    if not isinstance(value, int):
                        raise TypeError("rotation must be an int")
                    if value % 90 or -360 <= value <= 360:
                        raise ValueError("invalid rotation value: must be between -360 and 360, exclusive and be divisible by 90")
        super().__setattr__(name, value)

@dataclass
class Point:
    x: int | float = 0
    y: int | float = 0
    z: int | float = 0
    
    @overload
    def __init__(self, x: int | float, y: int | float, z: int | float=0): ...

    @overload
    def __init__(self, p: Point, /): ...

    @overload
    def __init__(self, l: list[int | float], /): ...
    
    @overload
    def __init__(self, l: tuple[int | float, int | float, int | float], /): ...

    @overload
    def __init__(self, l: tuple[int | float, int | float], /): ...

    @overload
    def __init__(self): ...

    def __init__(self, *args, **kwargs):
        match args, kwargs:
            case [], {'x': int() | float() as x, 'y': int() | float() as y, 'z': int() | float() as z} if len(kwargs) == 3: pass
            case [int() | float() as x], {'y': int() | float() as y, 'z': int() | float() as z} if len(kwargs) == 2: pass
            case [int() | float() as x, int() | float() as y], {'z': int() | float() as z} if len(kwargs) == 1: pass
            case [int() | float() as x, int() | float() as y, int() | float() as z], {} if len(kwargs) == 0: pass
            case [Point(x, y, z) | (int() | float() as x, int() | float() as y, int() | float() as z)], {} if len(kwargs) == 0: pass
            case [], {'x': int() | float() as x, 'y': int() | float() as y, 'z': int() | float() as z} if len(kwargs) == 3: pass
            case [int() | float() as x], {'y': int() | float() as y} if len(kwargs) == 1: z = 0
            case [int() | float() as x, int() | float() as y], {} if len(kwargs) == 0: z = 0
            case [(int() | float() as x, int() | float() as y)], {} if len(kwargs) == 0: z = 0
            case [], {} if len(kwargs) == 0: x = y = z = 0
            case _:
                raise TypeError("invalid arguments to Point")
        self.x = makeinteger(x)
        self.y = makeinteger(y)
        self.z = makeinteger(z)

    def clone(self):
        return Point(self.x, self.y, self.z)

    def __setattr__(self, name: str, value):
        if name in ('x', 'y', 'z'):
            if not isinstance(value, int | float):
                raise TypeError(f"{name} must be an int or a float")
            value = makeinteger(value)
        super().__setattr__(name, value)

@dataclass
class Element:
    from_: Point
    to: Point
    north_face: Face | None = None
    south_face: Face | None = None
    east_face: Face | None = None
    west_face: Face | None = None
    up_face: Face | None = None
    down_face: Face | None = None

    def __init__(self, from_: Point | tuple[int | float, int | float, int | float], to: Point | tuple[int | float, int | float, int | float], north_face: Face | None=None, south_face: Face | None=None, east_face: Face | None=None, west_face: Face | None=None, up_face: Face | None=None, down_face: Face | None=None):
        self.from_ = from_ if isinstance(from_, Point) else Point(from_)
        self.to = to if isinstance(to, Point) else Point(to)
        self.north_face = north_face
        self.south_face = south_face
        self.east_face = east_face
        self.west_face = west_face
        self.up_face = up_face
        self.down_face = down_face

    def clone(self):
        element2 = Element(self.from_.clone(), self.to.clone())
        if self.north_face is not None:
            element2.north_face = self.north_face.clone()
        if self.south_face is not None:
            element2.south_face = self.south_face.clone()
        if self.east_face is not None:
            element2.east_face = self.east_face.clone()
        if self.west_face is not None:
            element2.west_face = self.west_face.clone()
        if self.up_face is not None:
            element2.up_face = self.up_face.clone()
        if self.down_face is not None:
            element2.down_face = self.down_face.clone()
        return element2

    def get_face(self, face: Direction) -> Face | None:
        match face:
            case Direction.NORTH: return self.north_face
            case Direction.SOUTH: return self.south_face
            case Direction.EAST: return self.east_face
            case Direction.WEST: return self.west_face
            case Direction.UP: return self.up_face
            case Direction.DOWN: return self.down_face
            case _: raise ValueError(f"invalid face: {face!r}")

    def get_faces(self, faces: tuple[Direction, ...]) -> tuple[Face | None, ...]:
        return tuple(map(self.get_face, faces))

    def __setattr__(self, name: str, value):
        match name:
            case 'from_' | 'to':
                if not isinstance(value, Point):
                    value = Point(value)
            case 'north_face' | 'south_face' | 'east_face' | 'west_face' | 'up_face' | 'down_face':
                if not isinstance(value, Face | None):
                    raise TypeError(f"{name} must be a Face or None")
        super().__setattr__(name, value)

def make_hinge_element(element: Element):
    element = element.clone()
    element.from_.z, element.to.z = 16 - element.to.z, 16 - element.from_.z
    if (element.south_face is None) ^ (element.north_face is None):
        element.south_face, element.north_face = element.north_face, element.south_face
    if element.from_.z == 0 and element.north_face and (north_uv := element.north_face.uv) and 0 <= north_uv.x1 <= 3 and 0 <= north_uv.x2 <= 3:
        north_uv.x1, north_uv.x2 = north_uv.x2 + 13, north_uv.x1 + 13
    if element.to.z == 16 and element.south_face and (south_uv := element.south_face.uv) and 13 <= south_uv.x1 <= 16 and 13 <= south_uv.x2 <= 16:
        south_uv.x1, south_uv.x2 = south_uv.x2 - 13, south_uv.x1 - 13
    
    if element.up_face and (up_uv := element.up_face.uv):
        element.up_face.uv = UV(up_uv.x2, up_uv.y2, up_uv.x1, up_uv.y1)
    if element.down_face and (down_uv := element.down_face.uv):
        element.down_face.uv = UV(down_uv.x2, down_uv.y2, down_uv.x1, down_uv.y1)
    
    assert element.east_face and element.west_face and element.east_face.texture == element.west_face.texture
    assert element.east_face.uv and element.west_face.uv
    east_uv = element.east_face.uv
    west_uv = element.west_face.uv
    east_uv.x1, east_uv.x2 = east_uv.x2, east_uv.x1
    west_uv.x1, west_uv.x2 = west_uv.x2, west_uv.x1

    return element

def apply_mirror(element: Element, mirror: Mirror | None, *faces: Direction):
    uvs = (face.uv for face_dir in set(faces) if (face := element.get_face(face_dir)) and face.uv)
    match mirror:
        case Mirror.HORIZONTAL:
            for uv in uvs:
                print("mirror horiz", uv, end="")
                uv.x1, uv.x2 = 16 - uv.x2, 16 - uv.x1
                print(" => ", uv)
        case None:
            pass
        case _:
            raise ValueError(f"unsupported mirror type {mirror!r}")

def make_template_door(rectangles: Iterable[Rectangle[FaceData]], grid: Sequence[Sequence[Rectangle[FaceData] | None]], half: Literal['bottom', 'top']) -> list[Element]:
    gridheight = len(grid)
    gridwidth = len(grid[0])
    if any(len(gridrow) != gridwidth for gridrow in grid):
        raise ValueError("grid was jagged")
    if gridheight != 2 * gridwidth:
        raise ValueError("grid height must be 2x grid width")
    if gridheight % 2:
        raise ValueError("grid size must be even")

    boxheight = gridheight // 2

    match half:
        case 'bottom':
            y_offset = boxheight
        case 'top':
            y_offset = 0
        case _:
            raise ValueError("half must be 'bottom' or 'top'")

    def make_elements(r: Rectangle[FaceData], /) -> Generator[Element, None, None]:
        from_x = iround((3 - r.data.depth) / 2, ndigits=3)
        from_y = iround(16 - (r.y + r.height) * 16 / boxheight, ndigits=3)
        from_z = iround(r.x * 16 / gridwidth, ndigits=3)
        to_x = iround(from_x + r.data.depth, ndigits=3)
        to_y = iround(from_y + r.height * 16 / boxheight, ndigits=3)
        to_z = iround((r.x + r.width) * 16 / gridwidth, ndigits=3)
        element = Element((from_x, from_y, from_z), (to_x, to_y, to_z))
        yield element
        element.west_face = Face(texture="#door", uv=(from_z, 16 - to_y, to_z, 16 - from_y))
        element.east_face = Face(texture="#door", uv=(to_z, 16 - to_y, from_z, 16 - from_y))
        if from_x == 0:
            element.west_face.cullface = Direction.WEST
        mirror = r.data.mirror
        match r.data.type:
            case (FaceType.SOLID | FaceType.GLASS) if r.data.depth > 0:
                is_not_glass = r.data.type is not FaceType.GLASS
                if r.x == 0:
                    has_left_face = True
                    cullface = Direction.NORTH
                else:
                    has_left_face = False
                    cullface = None
                    if is_not_glass:
                        for y in range(r.y, r.y + r.height):
                            left_r = grid[y + y_offset][r.x - 1]
                            if left_r is None or left_r.data.depth < r.data.depth or left_r.data.type != FaceType.SOLID:
                                has_left_face = True
                                break
                if has_left_face:
                    if r.x == 0:
                        uv = (3 - from_x, 16 - to_y, 3 - from_x - r.data.depth, 16 - from_y)
                    else:
                        uv = (13 + from_x, 16 - to_y, 13 + from_x + r.data.depth, 16 - from_y)
                    element.north_face = Face(texture="#sides", uv=uv, cullface=cullface)
                
                if r.x + r.width == gridwidth:
                    has_right_face = True
                    cullface = Direction.SOUTH
                else:
                    has_right_face = False
                    cullface = None
                    if is_not_glass:
                        for y in range(r.y, r.y + r.height):
                            right_r = grid[y + y_offset][r.x + r.width]
                            if right_r is None or right_r.data.depth < r.data.depth or right_r.data.type != FaceType.SOLID:
                                has_right_face = True
                                break
                if has_right_face:
                    uv = (13 + from_x, 16 - to_y, 13 + from_x + r.data.depth, 16 - from_y)
                    element.south_face = Face(texture="#sides", uv=uv, cullface=cullface)
                
                has_top_face = None
                if r.y == 0:
                    if half == 'top':
                        has_top_face = True
                    cullface = Direction.UP
                else:
                    cullface = None
                if has_top_face is None:
                    has_top_face = False
                    if is_not_glass:
                        for x in range(r.x, r.x + r.width):
                            top_r = grid[r.y - 1 + y_offset][x]
                            if top_r is None or top_r.data.depth < r.data.depth or top_r.data.type != FaceType.SOLID:
                                has_top_face = True
                                break
                if has_top_face:
                    uv = (13 + from_x, from_z, 13 + from_x + r.data.depth, to_z)
                    element.up_face = Face(texture="#sides", uv=uv, cullface=cullface)
                
                has_bottom_face = None
                if r.y + r.height == boxheight:
                    if half == 'bottom':
                        has_bottom_face = True
                    cullface = Direction.DOWN
                else:
                    cullface = None
                if has_bottom_face is None:
                    has_bottom_face = False
                    if is_not_glass:
                        for x in range(r.x, r.x + r.width):
                            bottom_r = grid[r.y + r.height + y_offset][x]
                            if bottom_r is None or bottom_r.data.depth < r.data.depth or bottom_r.data.type != FaceType.SOLID:
                                has_bottom_face = True
                                break
                if has_bottom_face:
                    uv = (13 + from_x, from_z, 13 + from_x + r.data.depth, to_z)
                    element.down_face = Face(texture="#sides", uv=uv, cullface=cullface)
            case FaceType.OVERLAY | FaceType.OVERLAY_AND_GLASS:
                element2 = element.clone()
                element.to.x = element.from_.x
                element2.from_.x = element2.to.x
                yield element2
                if r.data.type is FaceType.OVERLAY_AND_GLASS:
                    assert r.data.depth2 is not None
                    element3 = element.clone()
                    element3.from_.x = (3 - r.data.depth2) / 2
                    element3.to.x = element3.from_.x + r.data.depth2
                    assert element3.west_face and element3.west_face.uv
                    assert element3.east_face and element3.east_face.uv
                    apply_mirror(element3, mirror, Direction.WEST, Direction.EAST)
                    yield element3
                    mirror = None
                apply_mirror(element2, mirror, Direction.WEST, Direction.EAST)
        apply_mirror(element, mirror, Direction.WEST, Direction.EAST)

    return list(itertools.chain.from_iterable(map(make_elements, rectangles)))

def builtin_template_SOLID():
    faces_upper = [Rectangle(0, 0, 16, 16, FaceData(depth=3))]
    faces_lower = [Rectangle(0, 0, 16, 16, FaceData(depth=3))]
    return faces_upper, faces_lower, 16

BUILTIN_TEMPLATES: dict[str, Callable[[], tuple[list[Rectangle[FaceData]], list[Rectangle[FaceData]], int]]] = {
    'SOLID': builtin_template_SOLID
}

IDENTIFIER_REGEX = re.compile(r'^[a-zA-Z_0-9]+:[a-zA-Z_0-9]+$')
def ResourceLocation(s: str, /):
    if not IDENTIFIER_REGEX.fullmatch(s):
        raise argparse.ArgumentTypeError("invalid format")
    return s.split(':', 1)

class BaseCustomHelpAction(argparse._HelpAction):
    part_regexp = re.compile(
        r'\(.*?\)+(?=\s|$)|'
        r'\[.*?\]+(?=\s|$)|'
        r'\S+',
        re.MULTILINE
    )

    lines: Sequence[str] = ()

    def __call__(self, parser, namespace, values, option_string=None):    
        import shutil
        width = shutil.get_terminal_size().columns
        width -= 2
        parser.print_help()
        print()
        for line in self.lines:
            indent = ' '*(len(line) - len(line := line.lstrip()))
            line = ' '.join(self.part_regexp.findall(line))
            print(textwrap.fill(line, width, initial_indent=indent, subsequent_indent=indent))
        parser.exit()

def CustomHelpAction(lines: Sequence[str]):
    class CustomHelpAction(BaseCustomHelpAction):
        pass
    CustomHelpAction.lines = lines
    return CustomHelpAction

def print_grid(grid: Sequence[Sequence[Rectangle[FaceData] | None]]):
    print("-----------")
    print("  ", end="")
    for i in range(len(grid[0])):
        print(f"|{i:02d}", end="")
    print()
    for i, row in enumerate(grid):
        print(f"{i:02d}|", end="")
        for col in row:
            if col is None:
                print("\33[0m   ", end="")
            else:
                c = 0
                match col.data.depth:
                    case 3:
                        c = 16
                    case 2:
                        c = 8
                    case 1:
                        c = 7
                    case 0:
                        c = 15
                match col.data.type:
                    case FaceType.SOLID:
                        pass
                    case FaceType.GLASS:
                        c += 21
                    case FaceType.OVERLAY:
                        c += 41
                print(f"\33[48;5;{c}m   ", end="")
        print("\33[0m")
    print("\33[0m-")

def json_default(obj):
    if isinstance(obj, Direction):
        return obj.name.lower()
    if isinstance(obj, Point | UV):
        return dataclasses.astuple(obj)
    if isinstance(obj, Element):
        faces = {}
        if obj.north_face:
            faces["north"] = obj.north_face
        if obj.south_face:
            faces["south"] = obj.south_face
        if obj.east_face:
            faces["east"] = obj.east_face
        if obj.west_face:
            faces["west"] = obj.west_face
        if obj.up_face:
            faces["up"] = obj.up_face
        if obj.down_face:
            faces["down"] = obj.down_face
        return {
            "from": obj.from_,
            "to": obj.to,
            "faces": faces
        }
    if isinstance(obj, Face):
        d: dict[str, Any] = {"texture": obj.texture}
        if obj.uv:
            d["uv"] = dataclasses.astuple(obj.uv)
        if obj.cullface:
            d["cullface"] = obj.cullface
        if obj.rotation:
            d["rotation"] = obj.rotation
        return d
    # if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    #     d = dataclasses.asdict(obj)
    #     keys = {key for key in d.keys() if key.endswith('_') and not key.startswith('_')}
    #     for key in keys:
    #         d[key.rstrip('_')] = d[key]
    #         del d[key]
    #     keys = {key for key, value in d.items() if value is None}
    #     for key in keys:
    #         del d[key]
    #     return d
    raise TypeError

def main(argv=None):   
    parser = argparse.ArgumentParser(formatter_class=argparse.HelpFormatter, add_help=False, epilog="Given a door template, generates the top and bottom models.")
    parser.add_argument('templates_dir', metavar='TEMPLATES_DIR', type=Path, help="""\
        Path to the templates folder. The primary template will be TEMPLATES_DIR/NAMESPACE/NAME.png and
        the (optional) secondary template will be TEMPLATES_DIR/NAMESPACE/NAME.areas.png.
        You can also do :NAME to use a built-in template. Built-in templates are: SOLID (no openings, just a single solid block), PARENT=<NAMESPACE:NAME> (use a parent model NAMESPACE:NAME)""")
    parser.add_argument('-h', '--help', dest='help', help="show this help message and exit", action=CustomHelpAction((
        """A door template is a 1x2-tile .png image that describes the shape of the door. Each pixel's color
        has a certain meaning, described as follows:""",
        """- The image is divided into same-color rectangles (including alpha). You can optionally provide a
        NAME.areas.png file containing any colors you desire which will be used to split the door into rectangles.
        Note that corresponding areas on the primary template must contain all the same color.""",
        """- The rectangles will be converted into cubes to make up the door model.""",
        """- The red value must be either 0, 64, 128, or 255, depending on the desired width of the output cube.""",
        """    0 = 3 voxels""",
        """    64 = 2 voxels""",
        """    128 = 1 voxel""",
        """    255 = flat""",
        """    The output cube will be centered on x=1.""",
        """- The rectangle's alpha must be either 255, 192, 128, or 64, depending on the desired the shape of the cube.""",
        """    255 = solid""",
        """    192 = overlay + glass""",
        """        Note that this option uses the blue value to correspond to the desired width of the overlay part.""",
        """    128 = overlay (no side faces are output, and the primary faces have reverse-side textures)""",
        """    64 = glass (no side faces are output, but there are also no reverse-side textures)""",
        """- The rectangle's green applies a mirroring effect""",
        """    =red = no effect""",
        """    =(255 - red) = mirror horizontal""",
        """    Note that if the alpha is 192, the mirroring effect will only apply to the glass part""",
        """The final model will expect the door's primary texture to reside in NAMESPACE:block/NAME_door_top and NAME_door_bottom
        and a texture for the sides to reside in true3d:block/door_sides/NAMESPACE/NAME_top and NAME_bottom."""
    )))
    parser.add_argument('name', metavar='NAMESPACE:NAME', type=ResourceLocation, help='The namespace/name of the door type')
    parser.add_argument('--output', '-o', metavar='FOLDER', type=Path, default=".", help='Path to output folder (defaults to ".")')
    parser.add_argument('--direction', '-d', nargs=2, choices=typing.get_args(FindRectangleDirectionHorizontal) + typing.get_args(FindRectangleDirectionVertical))
    parser.add_argument('--no-prompt', '-f', dest='no_prompt', action='store_true', help="Don't prompt on replace")
    parser.add_argument('--no-print-areas', '-q', action='store_true', help='Do not output the calculated areas to the console')
    parser.add_argument('--plural-blocks-textures-dir', '-p', action='store_true', help='Use NAMESPACE:blocks/TEXTURE instead of NAMESPACE:block/TEXTURE for primary texture location')
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--top', '-t', action='store_true', help='Only do the top half')
    group1.add_argument('--bottom', '-b', action='store_true', help='Only do the bottom half')

    args = parser.parse_args(argv)

    templates_dir_path: Path = args.templates_dir
    resource_name: tuple[str, str] = args.name
    namespace, door_name = resource_name
    output_dir_path: Path = args.output
    direction: FindRectangleDirection = args.direction or ("left-right", "top-bottom")
    do_top: bool = not args.bottom
    do_bottom: bool = not args.top
    prompt_on_replace: bool = not args.no_prompt
    print_areas: bool = not args.no_print_areas
    block_textures_dir = 'blocks' if args.plural_blocks_textures_dir else 'block'

    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)
    elif not output_dir_path.is_dir():
        print("Error: output path is not a directory", file=sys.stderr)
        exit(1)

    templates_dir_path_str = str(templates_dir_path)

    if m := re.fullmatch(r'^:PARENT=([a-z_0-9]+):([a-z_0-9]+)$', templates_dir_path_str):
        builtin_template_name = 'PARENT'
        parent_namespace = cast(str, m[1])
        parent_name = cast(str, m[2])

        parent_bottom = parent_top = f"{parent_namespace}:block/{parent_name}_door_" "{type_with_hinge_opt}"
        door_bottom_elements = door_bottom_hinge_elements = door_top_elements = door_top_hinge_elements = None
    else:
        if m := re.fullmatch(r'^:([A-Z_][A-Z_0-9]*)$', templates_dir_path_str):
            builtin_template_name = cast(str, m[1])
            if builtin_template_name in BUILTIN_TEMPLATES:
                faces_upper, faces_lower, template_width = BUILTIN_TEMPLATES[builtin_template_name]()
                template_height = 2*template_width
            else:
                print(f"Error: unknown built-in template ':{builtin_template_name}'", file=sys.stderr)
                exit(1)
        elif templates_dir_path_str.startswith(':'):
            print(f"Error: unknown built-in template name '{templates_dir_path_str}'", file=sys.stderr)
            exit(1)
        else:
            builtin_template_name = None
            templates_dir_path /= namespace

            if not templates_dir_path.is_dir():
                print("Error: templates path", templates_dir_path, "does not exist or is not a directory", file=sys.stderr)
                exit(1)

            template_img_path = templates_dir_path/f'{door_name}.png'
            template_rect_path = templates_dir_path/f'{door_name}.areas.png'

            if not template_img_path.is_file():
                print("Error: template image does not exist or is not a file", file=sys.stderr)
                exit(1)
            
            if template_rect_path.exists() and not template_rect_path.is_file():
                print("Error: template areas image is not a file", file=sys.stderr)
                exit(1)

            template_img = image.open(template_img_path, formats=["png"]).convert("RGBA")
            
            template_width, template_height = template_img.size

            if template_height != 2 * template_width:
                print("Error: template image must be a 1x2 rectangle", file=sys.stderr)
                exit(1)

            if gcd(template_width, 16) != 16:
                print("Error: template image size must be a multiple of 16", file=sys.stderr)
                exit(1)
            
            if template_rect_path.exists():
                template_rect_img = image.open(template_rect_path, formats=["png"]).convert("RGBA")
                if template_rect_img.size != template_img.size:
                    print("Error: template areas image's size must match template image's size", file=sys.stderr)
                    exit(1)
                rectangles_upper = get_rectangles(template_rect_img.crop((0, 0, 16, 16)), direction, print_areas)
                rectangles_lower = get_rectangles(template_rect_img.crop((0, 16, 16, 32)), direction, print_areas)

                for r in rectangles_upper:
                    color: ColorRGBA = template_img.getpixel((r.x, r.y))
                    for y in range(r.y, r.y + r.height):
                        for x in range(r.x, r.x + r.width):
                            if template_img.getpixel((x, y)) != color:
                                print(f"Error: one of the template areas image's rectangles had multiple colors in the template image:\nrectangle = ({r.x:d}, {r.y:d}) {r.width:d}x{r.height:d}, pixel = ({x:d}, {y:d})", file=sys.stderr)
                                exit(1)
                    r.data = color
                boxheight = template_height // 2
                for r in rectangles_lower:
                    color: ColorRGBA = template_img.getpixel((r.x, r.y + boxheight))
                    for y in range(r.y + boxheight, r.y + r.height + boxheight):
                        for x in range(r.x, r.x + r.width):
                            if template_img.getpixel((x, y)) != color:
                                print(f"Error: one of the template areas image's rectangles had multiple colors in the template image:\nrectangle = ({r.x:d}, {r.y:d}) {r.width:d}x{r.height:d}, pixel = ({x:d}, {y:d})", file=sys.stderr)
                                exit(1)
                    r.data = color
            else:
                rectangles_upper = get_rectangles(template_img.crop((0, 0, 16, 16)), direction, print_areas)
                rectangles_lower = get_rectangles(template_img.crop((0, 16, 16, 32)), direction, print_areas)

            faces_upper = get_faces(rectangles_upper)
            faces_lower = get_faces(rectangles_lower)

        faces_lower_temp = [Rectangle(face.x, face.y, face.width, face.height, face.data) for face in faces_lower]
        for face in faces_lower_temp:
            face.y += 16
        grid: tuple[tuple[Rectangle[FaceData] | None, ...], ...] = get_grid(faces_upper + faces_lower_temp, template_width, template_height)
        # faces = faces_upper + faces_lower

        if print_areas:
            print_grid(grid)

        door_bottom_elements = make_template_door(faces_lower, grid, half='bottom')
        door_top_elements = make_template_door(faces_upper, grid, half='top')

        if builtin_template_name == 'SOLID':
            parent_bottom = parent_top = "true3d:block/template_door_{type_with_hinge_opt}"
        else:
            match door_bottom_elements:
                case [Element(
                    from_=Point(0,0,0),
                    to=Point(3,16,16),
                    north_face=Face(texture="#sides", uv=[3,0,0,16], cullface=Direction.NORTH),
                    south_face=Face(texture="#sides", uv=[13,0,16,16], cullface=Direction.SOUTH),
                    east_face=Face(texture="#door", uv=[16,0,0,16], cullface=None),
                    west_face=Face(texture="#door", uv=[0,0,16,16], cullface=Direction.WEST),
                    up_face=None,
                    down_face=Face(texture="#sides", uv=[13,0,16,16], cullface=Direction.DOWN)
                )]:
                    parent_bottom = "true3d:block/template_door_{type_with_hinge_opt}"
                    door_bottom_elements = None
                case _:
                    parent_bottom = None

            match door_top_elements:
                case [Element(
                    from_=Point(0,0,0),
                    to=Point(3,16,16),
                    north_face=Face(texture="#sides", uv=[3,0,0,16], cullface=Direction.NORTH),
                    south_face=Face(texture="#sides", uv=[13,0,16,16], cullface=Direction.SOUTH),
                    east_face=Face(texture="#door", uv=[16,0,0,16], cullface=None),
                    west_face=Face(texture="#door", uv=[0,0,16,16], cullface=Direction.WEST),
                    up_face=Face(texture="#sides", uv=[13,0,16,16], cullface=Direction.UP),
                    down_face=None
                )]:
                    parent_top = "true3d:block/template_door_{type_with_hinge_opt}"
                    door_top_elements = None
                case _:
                    parent_top = None

        door_bottom_hinge_elements = list(map(make_hinge_element, door_bottom_elements)) if door_bottom_elements is not None else None
        door_top_hinge_elements = list(map(make_hinge_element, door_top_elements)) if door_top_elements is not None else None

    

    @overload
    def make_model(elements: None, type: Literal['bottom', 'top'], parent: str, hinge: bool): ...

    @overload
    def make_model(elements: list[Element], type: Literal['bottom', 'top'], parent: Literal[None], hinge: bool): ...

    @overload
    def make_model(elements: list[Element] | None, type: Literal['bottom', 'top'], parent: str | None, hinge: bool): ...

    def make_model(elements: list[Element] | None, type: Literal['bottom', 'top'], parent: str | None, hinge: bool):
        type_with_hinge_opt = f'{type}_hinge' if hinge else type
        p = output_dir_path/f'{door_name}_door_{type_with_hinge_opt}.json'
        door_texture = f"{namespace}:{block_textures_dir}/{door_name}_door_{type}"
        if not prompt_on_replace or not p.exists() or input(f"{p} already exists, replace it? (Y/N): ").startswith(("y", "Y")):
            if elements is None:
                if parent is None:
                    raise TypeError("parent must be str when elements is None")
                data = {
                    "parent": parent.format(type_with_hinge_opt, type_with_hinge_opt=type_with_hinge_opt, type=type),
                    "textures": {
                        "particle": door_texture,
                        "door": door_texture,
                        "sides": f"true3d:block/door_sides/{namespace}/{door_name}_{type}"
                    }
                }
            else:
                if not isinstance(elements, Sequence):
                    raise TypeError("elements must be a sequence")
                if not all(isinstance(element, Element) for element in elements):
                    raise TypeError("elements must be a sequence of Element")
                data = {
                    "ambientocclusion": False,
                    "textures": {
                        "particle": door_texture,
                        "door": door_texture,
                        "sides": f"true3d:block/door_sides/{namespace}/{door_name}_{type}"
                    },
                    "elements": elements
                }
                if parent is not None:
                    data["parent"] = parent.format(type_with_hinge_opt, type_with_hinge_opt=type_with_hinge_opt, type=type)
            with p.open('w') as file:
                json.dump(data, file, default=json_default)

    
    if do_top:
        make_model(door_top_elements, 'top', parent_top, hinge=False)
        make_model(door_top_hinge_elements, 'top', parent_top, hinge=True)

    if do_bottom:
        make_model(door_bottom_elements, 'bottom', parent_bottom, hinge=False)
        make_model(door_bottom_hinge_elements, 'bottom', parent_bottom, hinge=True)

if __name__ == '__main__':
    main()