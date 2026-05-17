#ifndef ISO_TRANSFORM_H
#define ISO_TRANSFORM_H

#include "raylib.h"

/* Convert a world-space tile coordinate to its screen-space pixel offset.
 * +X goes right-down on screen, +Y goes left-down; (0,0) maps to (0,0). */
Vector2 Iso_WorldToScreen(Vector2 world);

/* Inverse of Iso_WorldToScreen: convert a screen-space pixel offset
 * back to fractional tile coordinates. Round-trip safe for all inputs. */
Vector2 Iso_ScreenToWorld(Vector2 screen);

#endif
