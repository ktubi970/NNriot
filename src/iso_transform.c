#include "iso_transform.h"
#include "iso_config.h"

Vector2 Iso_WorldToScreen(Vector2 world)
{
    Vector2 screen;
    screen.x = (world.x - world.y) * (float)ISO_TILE_HALF_WIDTH;
    screen.y = (world.x + world.y) * (float)ISO_TILE_HALF_HEIGHT;
    return screen;
}

Vector2 Iso_ScreenToWorld(Vector2 screen)
{
    /* Inverse derivation absorbs the /2 of the average into the full
     * tile dimensions, so no extra divisor literal is needed:
     *   world.x = screen.x / W + screen.y / H
     *   world.y = screen.y / H - screen.x / W
     */
    Vector2 world;
    world.x = screen.x / (float)ISO_TILE_WIDTH  + screen.y / (float)ISO_TILE_HEIGHT;
    world.y = screen.y / (float)ISO_TILE_HEIGHT - screen.x / (float)ISO_TILE_WIDTH;
    return world;
}
