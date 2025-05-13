from typing import Annotated

from fastapi import APIRouter, Body
from fastapi.responses import HTMLResponse

from match3_demo.services import game_services

router = APIRouter(prefix="", tags=["Game"])


@router.get("/")
async def game():
    with open("src/match3_demo/index.html", mode="r") as f:
        html = f.read()
    return HTMLResponse(html)


@router.post("/{board_id}/reset")
async def reset(board_id: int):
    return game_services[board_id].reset()


@router.post("/{board_id}/swap")
async def swap(
    board_id: int, tile1: Annotated[list, Body()], tile2: Annotated[list, Body()]
):
    return game_services[board_id].swap(tile1, tile2)


@router.post("/{board_id}/swap-random")
async def swap(board_id: int):
    return game_services[board_id].swap_random()


@router.post("/{board_id}/swap-greedy")
async def swap(board_id: int):
    return game_services[board_id].swap_greedy()


@router.post("/{board_id}/swap-ppo")
async def swap(board_id: int):
    return game_services[board_id].swap_ppo()
