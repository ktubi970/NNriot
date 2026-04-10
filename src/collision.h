#ifndef COLLISION_H
#define COLLISION_H

#include <raylib.h>
#include "player.h"
#include "game_types.h"

// Radius (in world units) below which the helicopter collides with objects
#define COLLISION_HEIGHT_THRESHOLD 20.0f

// Radius of the player hitbox (world units)
#define PLAYER_COLLISION_RADIUS 18.0f

// Returns the solid collision radius for a given object type.
float GetObjectCollisionRadius(ObjectType objectType);

// Resolves player vs all world objects. Modifies player->position in place.
void Collision_ResolvePlayerVsObjects(Player *player,
                                      const Vector2 *positions,
                                      const ObjectType *types,
                                      int           count);

// Clamps player position to world bounds [0, worldSize].
void Collision_ClampToWorld(Player *player, float worldSize);

#endif
