#!python
from __future__ import annotations
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

from create_door_models import (
    iround,
    makeinteger,
    Rectangle,
    ColorRGBA,
    FaceType,
    Mirror,
    FaceData,
    FindRectangleDirection,
    FindRectangleDirectionHorizontal,
    FindRectangleDirectionVertical,
    get_rectangles,
    get_grid,
    Direction,
    UV,
    Face,
    Point,
    Element,
    apply_mirror,
    json_default,
    print_grid,
    CustomHelpAction,
    ResourceLocation
)

def face_data_from_color(color: ColorRGBA) -> FaceData:
    if color[3] == 0:
        raise ValueError("transparent color would result in no face")
    if color == (0, 0, 0, 255):
        return FaceData(depth=4)
    red, green, blue, alpha = color
    assert all(isinstance(c, int) for c in color), color
    assert 0 <= red <= 255, red
    assert 0 <= green <= 255, green
    assert 0 <= blue <= 255, blue
    assert 0 <= alpha <= 255, alpha
    d2 = None
    match red:
        case 0:
            d = 4
        case 64:
            d = 3
        case 128:
            d = 2
        case 192:
            d = 1
        case 255:
            d = 0
        case _:
            raise ValueError(f"invalid red value {red:d}, must be either 0, 64, 128, 192, or 255")
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
                    d2 = 4
                case 64:
                    d2 = 3
                case 128:
                    d2 = 2
                case 192:
                    d2 = 1
                case 255:
                    d2 = 0
                case _:
                    raise ValueError(f"invalid blue value {blue:d}, must be either 0, 64, 128, 192, or 255 when alpha value is 192")
        case 128:
            f = FaceType.OVERLAY
        case 64:
            f = FaceType.GLASS
        case _:
            raise ValueError(f"invalid alpha value {alpha:d}")

    return FaceData(depth=d, type=f, depth2=d2, mirror=m)

def get_faces(rectangles: Iterable[Rectangle[ColorRGBA]]) -> list[Rectangle[FaceData]]:
    faces: list[Rectangle[FaceData]] = []
    for r in rectangles:
        if r.data[3]:
            faces.append(Rectangle(r.x, r.y, r.width, r.height, face_data_from_color(r.data)))
    return faces

class TrapdoorStyle(Enum):
    SIMPLE = auto()
    WITH_SIDES = auto()
    WITH_UNIQUE_SIDES = auto()

def calc_left_face(r: Rectangle[FaceData], grid: Sequence[Sequence[Rectangle[FaceData] | None]], is_not_glass: bool, solid_cullface: Direction):
    if r.x == 0:
        has_left_face = True
        cullface = solid_cullface
    else:
        has_left_face = False
        cullface = None
        if is_not_glass:
            for y in range(r.y, r.y + r.height):
                left_r = grid[y][r.x - 1]
                if left_r is None or left_r.data.depth < r.data.depth or left_r.data.type != FaceType.SOLID:
                    has_left_face = True
                    break
    return has_left_face, cullface

def calc_right_face(r: Rectangle[FaceData], grid: Sequence[Sequence[Rectangle[FaceData] | None]], is_not_glass: bool, solid_cullface: Direction):
    gridwidth = len(grid[0])
    if r.x + r.width == gridwidth:
        has_right_face = True
        cullface = solid_cullface
    else:
        has_right_face = False
        cullface = None
        if is_not_glass:
            for y in range(r.y, r.y + r.height):
                right_r = grid[y][r.x + r.width]
                if right_r is None or right_r.data.depth < r.data.depth or right_r.data.type != FaceType.SOLID:
                    has_right_face = True
                    break
    return has_right_face, cullface

def calc_top_face(r: Rectangle[FaceData], grid: Sequence[Sequence[Rectangle[FaceData] | None]], is_not_glass: bool, solid_cullface: Direction):
    if r.y == 0:
        has_top_face = True
        cullface = solid_cullface
    else:
        has_top_face = False
        cullface = None
        if is_not_glass:
            for x in range(r.x, r.x + r.width):
                top_r = grid[r.y - 1][x]
                if top_r is None or top_r.data.depth < r.data.depth or top_r.data.type != FaceType.SOLID:
                    has_top_face = True
                    break
    return has_top_face, cullface

def calc_bottom_face(r: Rectangle[FaceData], grid: Sequence[Sequence[Rectangle[FaceData] | None]], is_not_glass: bool, solid_cullface: Direction):
    gridheight = len(grid)
    if r.y + r.height == gridheight:
        has_bottom_face = True
        cullface = solid_cullface
    else:
        has_bottom_face = False
        cullface = None
        if is_not_glass:
            for x in range(r.x, r.x + r.width):
                bottom_r = grid[r.y + r.height][x]
                if bottom_r is None or bottom_r.data.depth < r.data.depth or bottom_r.data.type != FaceType.SOLID:
                    has_bottom_face = True
                    break
    return has_bottom_face, cullface

def make_template_trapdoor_closed(rectangles: Iterable[Rectangle[FaceData]], grid: Sequence[Sequence[Rectangle[FaceData] | None]], half: Literal['bottom', 'top'], style: TrapdoorStyle) -> list[Element]:
    gridheight = len(grid)
    gridwidth = len(grid[0])
    if any(len(gridrow) != gridwidth for gridrow in grid):
        raise ValueError("grid was jagged")
    if gridheight != gridwidth:
        raise ValueError("grid must be square")

    match half:
        case 'bottom':
            y_offset = 13
        case 'top':
            y_offset = 0
        case _:
            raise ValueError("half must be 'bottom' or 'top'")
    
    def make_elements(r: Rectangle[FaceData], /) -> Generator[Element, None, None]:
        from_x = iround(16 - (r.x + r.width) * 16 / gridwidth, ndigits=3)
        from_y = iround((4 - r.data.depth) / 2, ndigits=3)
        from_z = iround(16 - r.y * 16 / gridheight, ndigits=3)
        to_x = iround(16 - r.x * 16 / gridwidth, ndigits=3)
        to_y = iround(from_y + r.data.depth, ndigits=3)
        to_z = iround(16 - (r.y + r.height) * 16 / gridheight, ndigits=3)
        element = Element((from_x, from_y + y_offset, from_z), (to_x, to_y + y_offset, to_z))
        yield element
        element.up_face = Face(texture="#face", uv=(16 - to_x, 16 - from_z, 16 - from_x, 16 - to_z), rotation=180)
        element.down_face = Face(texture="#face", uv=(16 - from_x, 16 - from_z, 16 - to_x, 16 - to_z))
        if element.to.y == 16:
            element.up_face.cullface = Direction.UP
        if element.from_.y == 0:
            element.down_face.cullface = Direction.DOWN
        mirror = r.data.mirror
        match r.data.type:
            case (FaceType.SOLID | FaceType.GLASS) if r.data.depth > 0:
                is_not_glass = r.data.type is not FaceType.GLASS
                has_left_face, cullface = calc_left_face(r, grid, is_not_glass, Direction.EAST)
                if has_left_face:
                    if to_x == 16:
                        match style:
                            case TrapdoorStyle.SIMPLE:
                                uv_y_offset = 0
                            case TrapdoorStyle.WITH_SIDES | TrapdoorStyle.WITH_UNIQUE_SIDES:
                                uv_y_offset = 10
                    else:
                        uv_y_offset = 3
                    element.east_face = Face(texture="#sides", uv=(16 - from_z, from_y + uv_y_offset, 16 - to_z, to_y + uv_y_offset), cullface=cullface)
                
                has_right_face, cullface = calc_right_face(r, grid, is_not_glass, Direction.WEST)
                if has_right_face:
                    if from_x == 0:
                        match style:
                            case TrapdoorStyle.SIMPLE:
                                uv_y_offset = 0
                            case TrapdoorStyle.WITH_SIDES:
                                uv_y_offset = 10
                            case TrapdoorStyle.WITH_UNIQUE_SIDES:
                                uv_y_offset = 7
                    else:
                        uv_y_offset = 3
                    element.west_face = Face(texture="#sides", uv=(from_z, from_y + uv_y_offset, to_z, to_y + uv_y_offset), cullface=cullface)
                
                has_top_face, cullface = calc_top_face(r, grid, is_not_glass, Direction.SOUTH)
                if has_top_face:
                    if to_z == 16:
                        uv_y_offset = 0
                    else:
                        uv_y_offset = 3
                    element.south_face = Face(texture="#sides", uv=(from_x, from_y + uv_y_offset, to_x, to_y + uv_y_offset), cullface=cullface)
                
                has_bottom_face, cullface = calc_bottom_face(r, grid, is_not_glass, Direction.NORTH)
                if has_bottom_face:
                    if from_z == 0:
                        uv_y_offset = 13
                    else:
                        uv_y_offset = 3
                    element.north_face = Face(texture="#sides", uv=(16 - from_x, from_y + uv_y_offset, 16 - to_x, to_y + uv_y_offset), cullface=cullface)
            case FaceType.OVERLAY | FaceType.OVERLAY_AND_GLASS:
                element2 = element.clone()
                element.to.y = element.from_.y
                element2.from_.y = element2.to.y
                yield element2
                if r.data.type is FaceType.OVERLAY_AND_GLASS:
                    assert r.data.depth2 is not None
                    element3 = element.clone()
                    element3.from_.y = (4 - r.data.depth2) / 2 + y_offset
                    element3.to.y = element3.from_.y + r.data.depth2
                    assert element3.up_face and element3.up_face.uv
                    assert element3.down_face and element3.down_face.uv
                    apply_mirror(element3, mirror, Direction.UP, Direction.DOWN)
                    yield element3
                    mirror = None
                apply_mirror(element2, mirror, Direction.UP, Direction.DOWN)
        apply_mirror(element, mirror, Direction.UP, Direction.DOWN)

    return list(itertools.chain.from_iterable(map(make_elements, rectangles)))

def make_template_trapdoor_open(rectangles: Iterable[Rectangle[FaceData]], grid: Sequence[Sequence[Rectangle[FaceData] | None]], style: TrapdoorStyle) -> list[Element]:
    gridheight = len(grid)
    gridwidth = len(grid[0])
    if any(len(gridrow) != gridwidth for gridrow in grid):
        raise ValueError("grid was jagged")
    if gridheight != gridwidth:
        raise ValueError("grid must be square")

    def make_elements(r: Rectangle[FaceData], /) -> Generator[Element, None, None]:
        from_x = iround(16 - (r.x + r.width) * 16 / gridwidth, ndigits=3)
        from_y = iround(r.y * 16 / gridheight, ndigits=3)
        from_z = 13 + iround((4 - r.data.depth) / 2, ndigits=3)
        to_x = iround(16 - r.x * 16 / gridwidth, ndigits=3)
        to_y = iround((r.y + r.height) * 16 / gridheight, ndigits=3)
        to_z = iround(from_z + r.data.depth, ndigits=3)
        element = Element((from_x, from_y, from_z), (to_x, to_y, to_z))
        yield element
        element.south_face = Face(texture="#face", uv=(16 - from_x, 16 - from_y, 16 - to_x, 16 - to_y))
        element.north_face = Face(texture="#face", uv=(16 - from_x, 16 - to_y, 16 - to_x, 16 - from_y), rotation=180)
        if element.to.z == 16:
            element.south_face.cullface = Direction.SOUTH
        if element.from_.z == 16:
            element.north_face.cullface = Direction.NORTH
        mirror = r.data.mirror
        match r.data.type:
            case (FaceType.SOLID | FaceType.GLASS) if r.data.depth > 0:
                is_not_glass = r.data.type is not FaceType.GLASS
                has_left_face, cullface = calc_left_face(r, grid, is_not_glass, Direction.EAST)
                if has_left_face:
                    if to_x == 16:
                        match style:
                            case TrapdoorStyle.SIMPLE:
                                uv_y_offset = 0
                            case TrapdoorStyle.WITH_SIDES | TrapdoorStyle.WITH_UNIQUE_SIDES:
                                uv_y_offset = 10
                    else:
                        uv_y_offset = 3
                    element.east_face = Face(texture="#sides", uv=(to_y, from_z - 13 + uv_y_offset, to_y, to_z - 13 + uv_y_offset), cullface=cullface, rotation=90)
                
                has_right_face, cullface = calc_right_face(r, grid, is_not_glass, Direction.WEST)
                if has_right_face:
                    if from_x == 0:
                        match style:
                            case TrapdoorStyle.SIMPLE:
                                uv_y_offset = 0
                            case TrapdoorStyle.WITH_SIDES:
                                uv_y_offset = 10
                            case TrapdoorStyle.WITH_UNIQUE_SIDES:
                                uv_y_offset = 7
                    else:
                        uv_y_offset = 3
                    element.west_face = Face(texture="#sides", uv=(from_y, from_z - 13 + uv_y_offset, to_y, to_z - 13 + uv_y_offset), cullface=cullface, rotation=90)
                
                has_top_face, cullface = calc_top_face(r, grid, is_not_glass, Direction.DOWN)
                if has_top_face:
                    if from_z == 0:
                        uv_y_offset = 0
                    else:
                        uv_y_offset = 3
                    element.down_face = Face(texture="#sides", uv=(from_x, from_z - 13 + uv_y_offset, to_x, to_z - 13 + uv_y_offset), cullface=cullface)
                
                has_bottom_face, cullface = calc_bottom_face(r, grid, is_not_glass, Direction.UP)
                if has_bottom_face:
                    if to_z == 16:
                        uv_y_offset = 13
                    else:
                        uv_y_offset = 3
                    element.up_face = Face(texture="#sides", uv=(16 - from_x, from_z - 13 + uv_y_offset, 16 - to_x, to_z - 13 + uv_y_offset), cullface=cullface)
            case FaceType.OVERLAY | FaceType.OVERLAY_AND_GLASS:
                element2 = element.clone()
                element.to.z = element.from_.z
                element2.from_.z = element2.to.z
                yield element2
                if r.data.type is FaceType.OVERLAY_AND_GLASS:
                    assert r.data.depth2 is not None
                    element3 = element.clone()
                    element3.from_.z = (4 - r.data.depth2) / 2
                    element3.to.z = element3.from_.z + r.data.depth2
                    assert element3.north_face and element3.north_face.uv
                    assert element3.south_face and element3.south_face.uv
                    apply_mirror(element3, mirror, Direction.NORTH, Direction.SOUTH)
                    yield element3
                    mirror = None
                apply_mirror(element2, mirror, Direction.NORTH, Direction.SOUTH)
        apply_mirror(element, mirror, Direction.NORTH, Direction.SOUTH)

    return list(itertools.chain.from_iterable(map(make_elements, rectangles)))

def main(argv=None):
    parser = argparse.ArgumentParser(add_help=True, epilog="Given a trapdoor template, generate the top, bottom, and open models")
    parser.add_argument('templates_dir', metavar='TEMPLATES_DIR', type=Path, help="""\
        Path to the templates folder. The primary template will be TEMPLATES_DIR/NAMESPACE/NAME.png and
        the (optional) secondary template will be TEMPLATES_DIR/NAMESPACE/NAME.areas.png.
        You can also do :NAME to use a built-in template. Built-in templates are: SOLID (no openings, just a single solid block), PARENT=<NAMESPACE:NAME> (use a parent model NAMESPACE:NAME)""")
    parser.add_argument('name', metavar='NAMESPACE:NAME', type=ResourceLocation, help='The namespace/name of the door type')
    parser.add_argument('--output', '-o', metavar='FOLDER', type=Path, default=".", help='Path to output folder (defaults to ".")')
    parser.add_argument('--direction', '-d', nargs=2, choices=typing.get_args(FindRectangleDirectionHorizontal) + typing.get_args(FindRectangleDirectionVertical))
    parser.add_argument('--no-prompt', '-f', dest='no_prompt', action='store_true', help="Don't prompt on replace")
    parser.add_argument('--no-print-areas', '-q', action='store_true', help='Do not output the calculated areas to the console')
    parser.add_argument('--plural-blocks-textures-dir', '-p', action='store_true', help='Use NAMESPACE:blocks/TEXTURE instead of NAMESPACE:block/TEXTURE for primary texture location')
    parser.add_argument('--style', '-s', choices=('simple', 'with-sides', 'with-unique-sides'), default='simple', help="The style of trapdoor. with-sides has unique side texture for the left and right faces, with-unique-sides has a unique texture for each side face.")
    
    args = parser.parse_args(argv)

    templates_dir_path: Path = args.templates_dir
    resource_name: tuple[str, str] = args.name
    namespace, door_name = resource_name
    output_dir_path: Path = args.output
    direction: FindRectangleDirection = args.direction or ("left-right", "top-bottom")
    prompt_on_replace: bool = not args.no_prompt
    print_areas: bool = not args.no_print_areas
    block_textures_dir = 'blocks' if args.plural_blocks_textures_dir else 'block'
    match args.style:
        case 'simple':
            style = TrapdoorStyle.SIMPLE
        case 'with-sides':
            style = TrapdoorStyle.WITH_SIDES
        case 'with-unique-sides':
            style = TrapdoorStyle.WITH_UNIQUE_SIDES
        case _:
            raise ValueError
    
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)
    elif not output_dir_path.is_dir():
        print("Error: output path is not a directory", file=sys.stderr)
        exit(1)

    templates_dir_path_str = str(templates_dir_path)

    