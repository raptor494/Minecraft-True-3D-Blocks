#!python
from __future__ import annotations
from enum import Enum, auto
import typing; from typing import Any, Callable, Iterable, Literal, NamedTuple, TypeAlias, Sequence, TypeVar, Generic, cast, overload
import dataclasses; from dataclasses import dataclass
from math import gcd
from json_utils import json
from PIL import Image as image
from PIL.Image import Image


if typing.TYPE_CHECKING:
    Number: TypeAlias = int | float
else:
    from numbers import Number

N = TypeVar('N', bound=Number)
T = TypeVar('T')

def makeinteger(val: int | float) -> int | float:
    if isinstance(val, float) and val.is_integer():
        return int(val)
    else:
        return val

@dataclass
class Rectangle(Generic[N, T]):
    x: N
    y: N
    width: N
    height: N
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

@dataclass
class FaceData:
    depth: Number = 3 # depth from 0 (exclusive) to 3
    type: FaceType = FaceType.SOLID

    @staticmethod
    def from_color(color: ColorRGBA) -> FaceData:
        if color[3] == 0:
            raise ValueError("transparent color would result in no face")
        if color[3] not in (255, 128, 64):
            raise ValueError(f"invalid alpha value {color[3]:d}")
        if color == (0, 0, 0, 255):
            return FaceData()
        if color[0] == color[1] == color[2]:
            match color[0]:
                case 0:
                    d = 3
                case 64:
                    d = 2
                case 128:
                    d = 1
                case 255:
                    d = 0
                case _:
                    raise ValueError(f"invalid shade {color[0]:d}, must be either 0, 64, 128, or 255")
            match color[3]:
                case 255:
                    f = FaceType.SOLID
                case 128:
                    f = FaceType.OVERLAY
                case 64:
                    f = FaceType.GLASS
                case _:
                    raise ValueError(f"invalid alpha value {color[3]:d}")
            return FaceData(depth=d, type=f)
        else:
            raise ValueError(f"invalid color {color[0]:02x}{color[1]:02x}{color[2]:02x}{color[3]:02x}")
        
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

def find_rectangle(img: Image, x: int, y: int, grid: Sequence[Sequence[Rectangle[int, ColorRGBA] | None]], direction: FindRectangleDirection=("left-right", "top-bottom")) -> Rectangle[int, ColorRGBA]:
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

def get_rectangles(img: Image, direction: FindRectangleDirection=("left-right", "top-bottom"), print_grid: bool=False) -> list[Rectangle[int, ColorRGBA]]:
    rectangles: list[Rectangle[int, ColorRGBA]] = []

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

    rectangle_grid: list[list[Rectangle[int, ColorRGBA] | None]] = [[None]*img.width for _ in range(img.height)]

    def add_rect(r: Rectangle[int, ColorRGBA], /):
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

def get_faces(rectangles: Iterable[Rectangle[int, ColorRGBA]]) -> list[Rectangle[int, FaceData]]:
    faces: list[Rectangle[int, FaceData]] = []
    for r in rectangles:
        if r.data[3]:
            faces.append(Rectangle(r.x, r.y, r.width, r.height, FaceData.from_color(r.data)))
    return faces

def get_grid(rectangles: Iterable[Rectangle[int, T]], width: int, height: int) -> tuple[tuple[Rectangle[int, T] | None, ...], ...]:
    grid: list[list[Rectangle[int, T] | None]] = [[None]*width for _ in range(height)]
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
class Face:
    texture: str
    uv: tuple[int | float, int | float, int | float, int | float] | None = None
    cullface: Direction | None = None

    def clone(self):
        return Face(self.texture, self.uv, self.cullface)

    def __setattr__(self, name: str, value):
        if name == 'uv':
            match value:
                case [int(x1) | float(x1), int(y1) | float(y1), int(x2) | float(x2), int(y2) | float(y2)]:
                    x1 = makeinteger(x1)
                    y1 = makeinteger(y1)
                    x2 = makeinteger(x2)
                    y2 = makeinteger(y2)
                    value = (x1, y1, x2, y2)
                case _:
                    raise TypeError("invalid uv value, must be sequence of 4 ints or floats")
        super().__setattr__(name, value)

@dataclass
class Point:
    x: int | float = 0
    y: int | float = 0
    z: int | float = 0

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
        if north_face is not None and not isinstance(north_face, Face):
            raise TypeError
        if south_face is not None and not isinstance(south_face, Face):
            raise TypeError
        if east_face is not None and not isinstance(east_face, Face):
            raise TypeError
        if west_face is not None and not isinstance(west_face, Face):
            raise TypeError
        if up_face is not None and not isinstance(up_face, Face):
            raise TypeError
        if down_face is not None and not isinstance(down_face, Face):
            raise TypeError
        self.north_face = north_face
        self.south_face = south_face
        self.east_face = east_face
        self.west_face = west_face
        self.up_face = up_face
        self.down_face = down_face

    def clone(self):
        element2 = Element(Point(self.from_), Point(self.to))
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

def make_hinge_element(element: Element):
    element = element.clone()
    element.from_.z, element.to.z = 16 - element.to.z, 16 - element.from_.z
    if (element.south_face is None) ^ (element.north_face is None):
        element.south_face, element.north_face = element.north_face, element.south_face
    if element.from_.z == 0 and element.north_face and element.north_face.uv and 0 <= element.north_face.uv[0] <= 3 and 0 <= element.north_face.uv[2] <= 3:
        element.north_face.uv = (element.north_face.uv[2] + 13, element.north_face.uv[1], element.north_face.uv[0] + 13, element.north_face.uv[3])
    if element.to.z == 16 and element.south_face and element.south_face.uv and 13 <= element.south_face.uv[0] <= 16 and 13 <= element.south_face.uv[2] <= 16:
        element.south_face.uv = (element.south_face.uv[2] - 13, element.south_face.uv[1], element.south_face.uv[0] - 13, element.south_face.uv[3])
    
    # if element.north_face and element.north_face.uv:
    #     element.north_face.uv = (element.north_face.uv[2], element.north_face.uv[1], element.north_face.uv[0], element.north_face.uv[3])
    # if element.south_face and element.south_face.uv:
    #     element.south_face.uv = (element.south_face.uv[2], element.south_face.uv[1], element.south_face.uv[0], element.south_face.uv[3])
    if element.up_face and element.up_face.uv:
        element.up_face.uv = (element.up_face.uv[2], element.up_face.uv[3], element.up_face.uv[0], element.up_face.uv[1])
    if element.down_face and element.down_face.uv:
        element.down_face.uv = (element.down_face.uv[2], element.down_face.uv[3], element.down_face.uv[0], element.down_face.uv[1])
    assert element.east_face and element.west_face and element.east_face.texture == element.west_face.texture
    assert element.east_face.uv and element.west_face.uv
    element.east_face.uv = (element.east_face.uv[2], element.east_face.uv[1], element.east_face.uv[0], element.east_face.uv[3])
    element.west_face.uv = (element.west_face.uv[2], element.west_face.uv[1], element.west_face.uv[0], element.west_face.uv[3])
    
    # element.east_face.uv, element.west_face.uv = element.west_face.uv, element.east_face.uv
    return element

def make_template_door_bottom(rectangles: Iterable[Rectangle[int, FaceData]], grid: Sequence[Sequence[Rectangle[int, FaceData] | None]]):
    gridheight = len(grid)
    gridwidth = len(grid[0])
    if gridheight != 2 * gridwidth:
        raise ValueError("grid height must be 2x grid width")
    if gridheight % 2:
        raise ValueError("grid size must be even")

    boxheight = gridheight // 2

    elements: list[Element] = []

    def add_element(r: Rectangle[int, FaceData]):
        from_x = (3 - r.data.depth) / 2
        from_y = 16 - (r.y + r.height) * 16 / boxheight
        from_z = r.x * 16 / gridwidth
        to_x = from_x + r.data.depth
        to_y = from_y + r.height * 16 / boxheight
        to_z = (r.x + r.width) * 16 / gridwidth
        element = Element((from_x, from_y, from_z), (to_x, to_y, to_z))
        element.west_face = Face(texture="#door", uv=(from_z, 16 - to_y, to_z, 16 - from_y))
        element.east_face = Face(texture="#door", uv=(to_z, 16 - to_y, from_z, 16 - from_y))
        if from_x == 0:
            element.west_face.cullface = Direction.WEST
        match r.data.type:
            case FaceType.SOLID:
                if r.x == 0:
                    has_left_face = True
                    cullface = Direction.NORTH
                    uv = (3 - from_x, 16 - to_y, 3 - from_x - r.data.depth, 16 - from_y)
                else:
                    has_left_face = False
                    cullface = None
                    uv = (13 + from_x, 16 - to_y, 13 + from_x + r.data.depth, 16 - from_y)
                    for y in range(r.y, r.y + r.height):
                        left_r = grid[y + boxheight][r.x - 1]
                        if left_r is None or left_r.data.depth < r.data.depth or left_r.data.type != FaceType.SOLID:
                            has_left_face = True
                            break
                if has_left_face:
                    element.north_face = Face(texture="#sides", uv=uv, cullface=cullface)
                if r.x + r.width == gridwidth:
                    has_right_face = True
                    cullface = Direction.SOUTH
                else:
                    has_right_face = False
                    cullface = None
                    for y in range(r.y, r.y + r.height):
                        try:
                            right_r = grid[y + boxheight][r.x + r.width]
                        except IndexError:
                            raise IndexError(f"{r.y = }, {r.height = }, {y = }, {y + boxheight = }, {r.x + r.width = }, {len(grid) = }, {len(grid[0]) = }")
                        if right_r is None or right_r.data.depth < r.data.depth or right_r.data.type != FaceType.SOLID:
                            has_right_face = True
                            break
                if has_right_face:
                    element.south_face = Face(texture="#sides", uv=(13 + from_x, 16 - to_y, 13 + from_x + r.data.depth, 16 - from_y), cullface=cullface)
                has_top_face = False
                if r.y == 0:
                    cullface = Direction.UP
                else:
                    cullface = None
                for x in range(r.x, r.x + r.width):
                    top_r = grid[r.y + boxheight - 1][x]
                    if top_r is None or top_r.data.depth < r.data.depth or top_r.data.type != FaceType.SOLID:
                        has_top_face = True
                        break
                if has_top_face:
                    element.up_face = Face(texture="#sides", uv=(13 + from_x, from_z, 13 + from_x + r.data.depth, to_z), cullface=cullface)
                if r.y + r.height == boxheight:
                    has_bottom_face = True
                    cullface = Direction.DOWN
                else:
                    has_bottom_face = False
                    cullface = None
                    for x in range(r.x, r.x + r.width):
                        try:
                            bottom_r = grid[r.y + r.height + boxheight][x]
                        except IndexError:
                            raise IndexError(f"{r.y = }, {r.height = }, {r.y + r.height + boxheight = }, {x = }, {len(grid) = }, {len(grid[0]) = }")
                        if bottom_r is None or bottom_r.data.depth < r.data.depth or bottom_r.data.type != FaceType.SOLID:
                            has_bottom_face = True
                            break
                if has_bottom_face:
                    # element.down_face = Face(texture="#sides", uv=(13 + from_x, 16 - elemwidth, r.data.depth, -elemwidth), cullface=cullface)
                    element.down_face = Face(texture="#sides", uv=(13 + from_x, from_z, 13 + from_x + r.data.depth, to_z), cullface=cullface)
            case FaceType.OVERLAY:
                element2 = element.clone()
                element.to.x = element.from_.x
                element2.from_.x = element2.to.x
                elements.append(element2)
            case FaceType.GLASS:
                pass
        elements.append(element)

    for r in rectangles:
        add_element(r)

    return elements

def make_template_door_bottom_hinge(rectangles: Iterable[Rectangle[int, FaceData]], grid: Sequence[Sequence[Rectangle[int, FaceData] | None]]):
    return list(map(make_hinge_element, make_template_door_bottom(rectangles, grid)))

def make_template_door_top(rectangles: Iterable[Rectangle[int, FaceData]], grid: Sequence[Sequence[Rectangle[int, FaceData] | None]]):
    gridheight = len(grid)
    gridwidth = len(grid[0])
    if gridheight != 2 * gridwidth:
        raise ValueError("grid height must be 2x grid width")
    if gridheight % 2:
        raise ValueError("grid size must be even")

    boxheight = gridheight // 2

    elements: list[Element] = []

    def add_element(r: Rectangle[int, FaceData]):
        from_x = (3 - r.data.depth) / 2
        from_y = 16 - (r.y + r.height) * 16 / boxheight
        from_z = r.x * 16 / gridwidth
        to_x = from_x + r.data.depth
        to_y = from_y + r.height * 16 / boxheight
        to_z = (r.x + r.width) * 16 / gridwidth
        element = Element((from_x, from_y, from_z), (to_x, to_y, to_z))
        element.west_face = Face(texture="#door", uv=(from_z, 16 - to_y, to_z, 16 - from_y))
        element.east_face = Face(texture="#door", uv=(to_z, 16 - to_y, from_z, 16 - from_y))
        # element.north_face = element.south_face = element.up_face = element.down_face = Face(texture="#sides")
        # elements.append(element)
        # return

        # elemwidth = r.width / gridwidth * 16
        # elemheight = r.height / boxheight * 16
        # from_x = (3 - r.data.depth) / 2
        # from_y = 16 - (r.y + r.height) / boxheight * 16
        # from_z = r.x / gridwidth * 16
        # to_x = from_x + r.data.depth
        # to_y = 16 - r.y / boxheight * 16
        # to_z = from_x + elemwidth
        # element = Element((from_x, from_y, from_z), (to_x, to_y, to_z))
        # element.west_face = Face(texture="#door", uv=(from_z, from_y, from_z + elemwidth, from_y + elemheight))
        if from_x == 0:
            element.west_face.cullface = Direction.WEST
        # element.east_face = Face(texture="#door", uv=(16 - from_z, from_y, 16 - from_z - elemwidth, from_y + elemheight))
        match r.data.type:
            case FaceType.SOLID:
                if r.x == 0:
                    has_left_face = True
                    cullface = Direction.NORTH
                    uv = (3 - from_x, 16 - to_y, 3 - from_x - r.data.depth, 16 - from_y)
                else:
                    has_left_face = False
                    cullface = None
                    uv = (13 + from_x, 16 - to_y, 13 + from_x + r.data.depth, 16 - from_y)
                    for y in range(r.y, r.y + r.height):
                        left_r = grid[y][r.x - 1]
                        if left_r is None or left_r.data.depth < r.data.depth or left_r.data.type != FaceType.SOLID:
                            has_left_face = True
                            break
                if has_left_face:
                    element.north_face = Face(texture="#sides", uv=uv, cullface=cullface)
                if r.x + r.width == gridwidth:
                    has_right_face = True
                    cullface = Direction.SOUTH
                else:
                    has_right_face = False
                    cullface = None
                    for y in range(r.y, r.y + r.height):
                        right_r = grid[y][r.x + r.width]
                        if right_r is None or right_r.data.depth < r.data.depth or right_r.data.type != FaceType.SOLID:
                            has_right_face = True
                            break
                if has_right_face:
                    element.south_face = Face(texture="#sides", uv=(13 + from_x, 16 - to_y, 13 + from_x + r.data.depth, 16 - from_y), cullface=cullface)
                if r.y == 0:
                    has_top_face = True
                    cullface = Direction.UP
                else:
                    has_top_face = False
                    cullface = None
                    for x in range(r.x, r.x + r.width):
                        top_r = grid[r.y - 1][x]
                        if top_r is None or top_r.data.depth < r.data.depth or top_r.data.type != FaceType.SOLID:
                            has_top_face = True
                            break
                if has_top_face:
                    element.up_face = Face(texture="#sides", uv=(13 + from_x, from_z, 13 + from_x + r.data.depth, to_z), cullface=cullface)
                has_bottom_face = False
                if r.y + r.height == boxheight:
                    cullface = Direction.DOWN
                else:
                    cullface = None
                for x in range(r.x, r.x + r.width):
                    bottom_r = grid[r.y + r.height][x]
                    if bottom_r is None or bottom_r.data.depth < r.data.depth or bottom_r.data.type != FaceType.SOLID:
                        has_bottom_face = True
                        break
                if has_bottom_face:
                    # element.down_face = Face(texture="#sides", uv=(13 + from_x, 16 - elemwidth, r.data.depth, -elemwidth), cullface=cullface)
                    element.down_face = Face(texture="#sides", uv=(13 + from_x, from_z, 13 + from_x + r.data.depth, to_z), cullface=cullface)
            case FaceType.OVERLAY:
                element2 = element.clone()
                element.to.x = element.from_.x
                element2.from_.x = element2.to.x
                elements.append(element2)
            case FaceType.GLASS:
                pass
        elements.append(element)

    for r in rectangles:
        add_element(r)

    return elements

def make_template_door_top_hinge(rectangles: Iterable[Rectangle[int, FaceData]], grid: Sequence[Sequence[Rectangle[int, FaceData] | None]]):
    return list(map(make_hinge_element, make_template_door_top(rectangles, grid)))

def builtin_template_SOLID():
    faces_upper = [Rectangle(0, 0, 16, 16, FaceData())]
    faces_lower = [Rectangle(0, 0, 16, 16, FaceData())]
    return faces_upper, faces_lower, 16

BUILTIN_TEMPLATES: dict[str, Callable[[], tuple[list[Rectangle[int, FaceData]], list[Rectangle[int, FaceData]], int]]] = {
    'SOLID': builtin_template_SOLID
}

def main(argv=None):
    from pathlib import Path
    import re
    import argparse
    import sys
    
    IDENTIFIER_REGEX = re.compile(r'^[a-zA-Z_0-9]+:[a-zA-Z_0-9]+$')
    def ResourceLocation(s: str, /):
        if not IDENTIFIER_REGEX.fullmatch(s):
            raise argparse.ArgumentTypeError("invalid format")
        return s.split(':', 1)
   
    class CustomHelpAction(argparse._HelpAction):
        def __call__(self, parser, namespace, values, option_string=None):
            import re
            part_regexp = re.compile(
                r'\(.*?\)+(?=\s|$)|'
                r'\[.*?\]+(?=\s|$)|'
                r'\S+',
                re.MULTILINE
            )
            import shutil
            width = shutil.get_terminal_size().columns
            width -= 2
            import textwrap
            lines = (
                """A door template is a 1x2-tile .png image that describes the shape of the door. Each pixel's color
                has a certain meaning, described as follows:""",
                """- The image is divided into same-color rectangles (including alpha). You can optionally provide a
                NAME.areas.png file containing any colors you desire which will be used to split the door into rectangles.
                Note that corresponding areas on the primary template must contain all the same color.""",
                """- The rectangles will be converted into cubes to make up the door model.""",
                """- All rectangles must be grayscale (that is, R = G = B)""",
                """- The grayscale value must be either 0, 64, 128, or 255, depending on the desired width of the output cube.""",
                """    0 = 3 voxels""",
                """    64 = 2 voxels""",
                """    128 = 1 voxel""",
                """    255 = flat""",
                """    The output cube will be centered on x=1.""",
                """- The rectangle's alpha must be either 255, 128, or 64, depending on the desired the shape of the cube.""",
                """    255 = solid""",
                """    128 = overlayd (no side faces are output, and the primary faces have reverse-side textures)""",
                """    64 = glass (no side faces are output, but there are also no reverse-side textures)""",
                """The final model will expect the door's primary texture to reside in NAMESPACE:block/NAME_door_top and NAME_door_bottom
                and a texture for the sides to reside in true3d:block/door_sides/NAMESPACE/NAME_top and NAME_bottom."""
            )
            parser.print_help()
            print()
            for line in lines:
                indent = ' '*(len(line) - len(line := line.lstrip()))
                line = ' '.join(part_regexp.findall(line))
                print(textwrap.fill(line, width, initial_indent=indent, subsequent_indent=indent))
            parser.exit()

    parser = argparse.ArgumentParser(formatter_class=argparse.HelpFormatter, add_help=False, epilog="Given a door template, generates the top and bottom models.")
    parser.add_argument('templates_dir', metavar='TEMPLATES_DIR', type=Path, help="""\
        Path to the templates folder. The primary template will be TEMPLATES_DIR/NAMESPACE/NAME.png and
        the (optional) secondary template will be TEMPLATES_DIR/NAMESPACE/NAME.areas.png.
        You can also do :NAME to use a built-in template. Built-in templates are: SOLID (no openings, just a single solid block)""")
    parser.add_argument('name', metavar='NAMESPACE:NAME', type=ResourceLocation, help='The namespace/name of the door type')
    parser.add_argument('-h', '--help', action=CustomHelpAction, dest='help', help="show this help message and exit")
    parser.add_argument('--output', '-o', metavar='FOLDER', type=Path, default=".", help='Path to output folder (defaults to ".")')
    parser.add_argument('--direction', '-d', nargs=2, choices=typing.get_args(FindRectangleDirectionHorizontal) + typing.get_args(FindRectangleDirectionVertical))
    parser.add_argument('--no-prompt', '-f', dest='no_prompt', action='store_true', help="Don't prompt on replace")
    parser.add_argument('--no-print-areas', '-q', action='store_true', help='Do not output the calculated areas to the console')
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

    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)
    elif not output_dir_path.is_dir():
        print("Error: output path is not a directory", file=sys.stderr)
        exit(1)

    if m := re.fullmatch(r'^:([A-Z_][A-Z_0-9]*)$', str(templates_dir_path)):
        builtin_template_name = cast(str, m[1])
        if builtin_template_name in BUILTIN_TEMPLATES:
            faces_upper, faces_lower, template_width = BUILTIN_TEMPLATES[builtin_template_name]()
            template_height = 2*template_width
        else:
            print(f"Error: unknown built-in template ':{builtin_template_name}'", file=sys.stderr)
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
    grid = get_grid(faces_upper + faces_lower_temp, template_width, template_height)
    # faces = faces_upper + faces_lower

    if print_areas:
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

    door_bottom_elements = make_template_door_bottom(faces_lower, grid)
    door_top_elements = make_template_door_top(faces_upper, grid)

    if builtin_template_name == 'SOLID':
        solid_bottom = solid_top = True
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
                solid_bottom = True
            case _:
                solid_bottom = False

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
                solid_top = True
            case _:
                solid_top = False

    door_bottom_hinge_elements = list(map(make_hinge_element, door_bottom_elements)) if not solid_bottom else None
    door_top_hinge_elements = list(map(make_hinge_element, door_top_elements)) if not solid_top else None

    def json_default(obj):
        if isinstance(obj, Direction):
            return obj.name.lower()
        if isinstance(obj, Point):
            return [obj.x, obj.y, obj.z]
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
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            d = dataclasses.asdict(obj)
            keys = {key for key in d.keys() if key.endswith('_') and not key.startswith('_')}
            for key in keys:
                d[key.rstrip('_')] = d[key]
                del d[key]
            keys = {key for key, value in d.items() if value is None}
            for key in keys:
                del d[key]
            return d
        raise TypeError

    @overload
    def make_model(elements: Any, type: Literal['bottom', 'top'], solid: Literal[True], hinge: bool): ...

    @overload
    def make_model(elements: list[Element], type: Literal['bottom', 'top'], solid: Literal[False], hinge: bool): ...

    @overload
    def make_model(elements: list[Element] | Any, type: Literal['bottom', 'top'], solid: bool, hinge: bool): ...

    def make_model(elements: list[Element], type: Literal['bottom', 'top'], solid: bool, hinge: bool):
        type_with_hinge_opt = f'{type}_hinge' if hinge else type
        p = output_dir_path/f'{door_name}_door_{type_with_hinge_opt}.json'
        if not prompt_on_replace or not p.exists() or input(f"{p} already exists, replace it? (Y/N): ").startswith(("y", "Y")):
            if solid:
                data = {
                    "parent": f"true3d:block/template_door_{type_with_hinge_opt}",
                    "textures": {
                        "door": f"{namespace}:block/{door_name}_door_{type}",
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
                        "particle": "#door",
                        "door": f"{namespace}:block/{door_name}_door_{type}",
                        "sides": f"true3d:block/door_sides/{namespace}/{door_name}_{type}"
                    },
                    "elements": elements
                }
            with p.open('w') as file:
                json.dump(data, file, default=json_default)

    
    if do_top:
        make_model(door_top_elements, 'top', solid_top, hinge=False)
        make_model(door_top_hinge_elements, 'top', solid_top, hinge=True)

    if do_bottom:
        make_model(door_bottom_elements, 'bottom', solid_bottom, hinge=False)
        make_model(door_bottom_hinge_elements, 'bottom', solid_bottom, hinge=True)

if __name__ == '__main__':
    main()