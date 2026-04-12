#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "collision.h"
#include "game_types.h"
#include "player.h"

int tests_run = 0;
int tests_failed = 0;

#define ASSERT(condition, message) \
    do { \
        tests_run++; \
        if (!(condition)) { \
            printf("[FAIL] %s\n", message); \
            tests_failed++; \
        } \
    } while (0)

void test_radii() {
    printf("Running test_radii...\n");
    ASSERT(GetObjectCollisionRadius(OBJ_HOTEL) == 60.0f, "Hotel radius should be 60");
    ASSERT(GetObjectCollisionRadius(OBJ_PALM) == 0.0f, "Palm radius should be 0 (passthrough)");
    ASSERT(GetObjectCollisionRadius(OBJ_ROCK) == 20.0f, "Rock radius should be 20");
}

void test_clamping() {
    printf("Running test_clamping...\n");
    Player p = { .position = { -10, 3100 } };
    Collision_ClampToWorld(&p, 3000.0f);
    ASSERT(p.position.x == 0.0f, "Should clamp negative X to 0");
    ASSERT(p.position.y == 3000.0f, "Should clamp overflow Y to 3000");
}

void test_collision_resolution() {
    printf("Running test_collision_resolution...\n");
    
    // Case 1: Low altitude collision
    Player p1 = { .position = { 100, 100 }, .height = 10.0f };
    Vector2 obs_pos = { 110, 100 }; // 10 units away
    ObjectType type = OBJ_ROCK; // Radius 20
    // Total min distance = 18 (player) + 20 (rock) = 38
    // Current distance = 10. Overlap = 28.
    
    Collision_ResolvePlayerVsObjects(&p1, &obs_pos, &type, 1);
    
    // Should be pushed back exactly to dist 38
    float final_dist = sqrtf(powf(p1.position.x - obs_pos.x, 2) + powf(p1.position.y - obs_pos.y, 2));
    ASSERT(final_dist >= 37.9f && final_dist <= 38.1f, "Player should be pushed outside collision radius at low altitude");

    // Case 2: High altitude (no collision)
    Player p2 = { .position = { 100, 100 }, .height = 50.0f };
    Collision_ResolvePlayerVsObjects(&p2, &obs_pos, &type, 1);
    ASSERT(p2.position.x == 100.0f && p2.position.y == 100.0f, "No collision should occur at high altitude");

    // Case 3: Passthrough object (Palm)
    Player p3 = { .position = { 100, 100 }, .height = 10.0f };
    ObjectType type_palm = OBJ_PALM;
    Collision_ResolvePlayerVsObjects(&p3, &obs_pos, &type_palm, 1);
    ASSERT(p3.position.x == 100.0f && p3.position.y == 100.0f, "Palm should not cause collision");
}

int main() {
    printf("--- DESERT STRIKE COLLISION TESTS ---\n");
    
    test_radii();
    test_clamping();
    test_collision_resolution();
    
    printf("\n-------------------------------------\n");
    printf("Tests Run: %d\n", tests_run);
    printf("Tests Failed: %d\n", tests_failed);
    
    return (tests_failed == 0) ? 0 : 1;
}
