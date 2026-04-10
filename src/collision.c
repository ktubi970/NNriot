#include "collision.h"
#include <math.h>

// Manual vector math since raymath.h might be missing in the include path
static float Vector2Dist(Vector2 v1, Vector2 v2) {
    float dx = v2.x - v1.x;
    float dy = v2.y - v1.y;
    return sqrtf(dx*dx + dy*dy);
}

float GetObjectCollisionRadius(ObjectType objectType) {
    switch (objectType) {
        case OBJ_PALM:         return 0.0f;
        case OBJ_ROCK:         return 20.0f;
        case OBJ_HUT:          return 28.0f;
        case OBJ_TOWER:        return 35.0f;
        case OBJ_HOTEL:        return 60.0f;
        case OBJ_CLUTTER_ROCK: return 0.0f;
        case OBJ_FACTORY:      return 55.0f;
        case OBJ_SILO:         return 30.0f;
        case OBJ_REFINERY:     return 50.0f;
        case OBJ_DOME:         return 45.0f;
        case OBJ_RADAR:        return 25.0f;
        case OBJ_SOLDIER:      return 0.0f;
        default:               return 0.0f;
    }
}

void Collision_ResolvePlayerVsObjects(Player *player,
                                      const Vector2 *positions,
                                      const ObjectType *types,
                                      int           count) 
{
    if (player->height >= COLLISION_HEIGHT_THRESHOLD) return;

    for (int i = 0; i < count; i++) {
        float objRadius = GetObjectCollisionRadius(types[i]);
        if (objRadius <= 0.0f) continue;

        float minDistance = PLAYER_COLLISION_RADIUS + objRadius;
        float dist = Vector2Dist(player->position, positions[i]);

        if (dist < minDistance) {
            float overlap = minDistance - dist;
            float pushX, pushY;

            if (dist > 0.001f) {
                // Direction from object to player
                pushX = (player->position.x - positions[i].x) / dist;
                pushY = (player->position.y - positions[i].y) / dist;
            } else {
                // Perfect overlap: push North
                pushX = 0.0f;
                pushY = -1.0f;
            }

            // Apply push-back
            player->position.x += pushX * overlap;
            player->position.y += pushY * overlap;
        }
    }
}

void Collision_ClampToWorld(Player *player, float worldSize) {
    if (player->position.x < 0) player->position.x = 0;
    if (player->position.y < 0) player->position.y = 0;
    if (player->position.x > worldSize) player->position.x = worldSize;
    if (player->position.y > worldSize) player->position.y = worldSize;
}
