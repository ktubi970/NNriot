#include <raylib.h>
#include <stdlib.h>
#include "player.h"
#include "resources.h"
#include "asset_metadata.h"
#include "collision.h"
#include <math.h>

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600
#define WORLD_SIZE 3000
#define MAX_OBJECTS 1000

#include "game_types.h"

typedef struct {
    Vector2 position; // World Top-down (X, Y)
    ObjectType type;
    float screenY;    // Calculation for depth sorting
    float scale;      // Randomized scale for variety
} WorldObject;

// For sorting the draw order
typedef struct {
    int id;           // -1 for Player, >=0 for WorldObject index
    float sortY;
} DrawEntry;

int CompareDrawEntries(const void* a, const void* b) {
    DrawEntry* eA = (DrawEntry*)a;
    DrawEntry* eB = (DrawEntry*)b;
    if (eA->sortY < eB->sortY) return -1;
    if (eA->sortY > eB->sortY) return 1;
    return 0;
}

Vector2 GetIsometricPos(Vector2 worldPos) {
    return (Vector2){ 
        (worldPos.x - worldPos.y),
        (worldPos.x + worldPos.y) / 2.0f 
    };
}

Vector2 GetWorldPos(Vector2 isoPos) {
    return (Vector2){
        isoPos.y + (isoPos.x / 2.0f),
        isoPos.y - (isoPos.x / 2.0f)
    };
}

void DrawGround(Texture2D sandTile, Camera2D camera) {
    // Determine visible isometric bounds
    Vector2 topLeftArr = GetScreenToWorld2D((Vector2){ 0, 0 }, camera);
    Vector2 bottomRightArr = GetScreenToWorld2D((Vector2){ SCREEN_WIDTH, SCREEN_HEIGHT }, camera);
    Vector2 topRightArr = GetScreenToWorld2D((Vector2){ SCREEN_WIDTH, 0 }, camera);
    Vector2 bottomLeftArr = GetScreenToWorld2D((Vector2){ 0, SCREEN_HEIGHT }, camera);

    // Convert all 4 isometric corners to world-space to find the bounding box
    Vector2 wTL = GetWorldPos(topLeftArr);
    Vector2 wBR = GetWorldPos(bottomRightArr);
    Vector2 wTR = GetWorldPos(topRightArr);
    Vector2 wBL = GetWorldPos(bottomLeftArr);

    float minX = fminf(fminf(wTL.x, wBR.x), fminf(wTR.x, wBL.x)) - 200;
    float maxX = fmaxf(fmaxf(wTL.x, wBR.x), fmaxf(wTR.x, wBL.x)) + 200;
    float minY = fminf(fminf(wTL.y, wBR.y), fminf(wTR.y, wBL.y)) - 200;
    float maxY = fmaxf(fmaxf(wTL.y, wBR.y), fmaxf(wTR.y, wBL.y)) + 200;

    // Clamp to world size (optional but safer)
    if (minX < 0) minX = 0;
    if (minY < 0) minY = 0;
    if (maxX > WORLD_SIZE) maxX = WORLD_SIZE;
    if (maxY > WORLD_SIZE) maxY = WORLD_SIZE;

    // Tile the sand texture in World-Space (which then gets projected isometrically)
    int tileSize = sandTile.width; 
    int startX = ((int)minX / tileSize) * tileSize;
    int startY = ((int)minY / tileSize) * tileSize;

    for (int x = startX; x < maxX; x += tileSize) {
        for (int y = startY; y < maxY; y += tileSize) {
            Vector2 p = GetIsometricPos((Vector2){ (float)x, (float)y });
            // Draw regular top-down tile (it looks "flatter" but consistent in iso mode)
            DrawTextureEx(sandTile, p, 0.0f, 1.0f, WHITE);
        }
    }
}

#include <time.h>

int main(void) {
    srand(time(NULL));
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Desert Strike: Theater Recon [ELEVATION & DEPTH]");
    SetTargetFPS(60);

    Resources res;
    if (!Resources_Load(&res)) return 1;

    Player player;
    Player_Init(&player);

    // Populate a diverse Desert World
    WorldObject objects[MAX_OBJECTS] = { 0 };
    int objCount = 0;
    
    // Add specific structures
    objects[objCount++] = (WorldObject){ { 1100, 1100 }, OBJ_PALM, 0, 1.0f };
    objects[objCount++] = (WorldObject){ { 950, 1200 }, OBJ_ROCK, 0, 1.2f };
    objects[objCount++] = (WorldObject){ { 1500, 1500 }, OBJ_HOTEL, 0, 1.0f };
    objects[objCount++] = (WorldObject){ { 800, 800 }, OBJ_TOWER, 0, 1.0f };
    objects[objCount++] = (WorldObject){ { 1200, 1400 }, OBJ_HUT, 0, 1.0f };
    objects[objCount++] = (WorldObject){ { 1250, 1420 }, OBJ_HUT, 0, 1.0f };
    objects[objCount++] = (WorldObject){ { 1000, 1500 }, OBJ_ROCK, 0, 0.9f };
    objects[objCount++] = (WorldObject){ { 1600, 900 }, OBJ_HOTEL, 0, 1.0f };
    
    // Add new Tactical Objectives
    objects[objCount++] = (WorldObject){ { 2100, 1500 }, OBJ_FACTORY, 0, 1.0f };
    objects[objCount++] = (WorldObject){ { 2300, 1600 }, OBJ_REFINERY, 0, 1.0f };
    objects[objCount++] = (WorldObject){ { 2400, 1400 }, OBJ_SILO, 0, 1.0f };
    objects[objCount++] = (WorldObject){ { 1900, 1200 }, OBJ_DOME, 0, 1.0f };
    
    // Add some soldiers
    for (int i = 0; i < 10; i++) {
        objects[objCount++] = (WorldObject){ { 2200 + (float)(rand()%200), 1550 + (float)(rand()%100) }, OBJ_SOLDIER, 0, 1.0f };
    }
    
    // Add MASSIVE random scatter (Clutter pass)
    while (objCount < MAX_OBJECTS) {
        float x = (float)(rand() % WORLD_SIZE);
        float y = (float)(rand() % WORLD_SIZE);
        
        // Randomly pick a clutter type
        int r = rand() % 10;
        ObjectType type = (r < 7) ? OBJ_CLUTTER_ROCK : (r < 9 ? OBJ_PALM : OBJ_ROCK);
        float scale = 0.5f + (float)(rand() % 100) / 100.0f;
        
        objects[objCount++] = (WorldObject){ { x, y }, type, 0, scale };
    }

    Camera2D camera = { 0 };
    camera.target = GetIsometricPos(player.position);
    camera.offset = (Vector2){ SCREEN_WIDTH / 2.0f, SCREEN_HEIGHT / 2.0f };
    camera.zoom = 1.0f;

    bool debugMode = false;

    while (!WindowShouldClose()) {
        float deltaTime = GetFrameTime();
        if (IsKeyPressed(KEY_F1)) debugMode = !debugMode;
        
        Player_Update(&player, deltaTime);

        // Resolve collisions
        {
            Vector2 positions[MAX_OBJECTS];
            ObjectType types[MAX_OBJECTS];
            for (int i = 0; i < objCount; i++) {
                positions[i] = objects[i].position;
                types[i] = objects[i].type;
            }
            Collision_ResolvePlayerVsObjects(&player, positions, types, objCount);
            Collision_ClampToWorld(&player, WORLD_SIZE);
        }

        Vector2 isoPlayer = GetIsometricPos(player.position);
        camera.target.x += (isoPlayer.x - camera.target.x) * 0.1f;
        camera.target.y += (isoPlayer.y - camera.target.y) * 0.1f;

        // Prepare Depth Sorting
        DrawEntry drawList[MAX_OBJECTS + 1];
        int drawCount = 0;
        
        // Add player to draw list
        drawList[drawCount++] = (DrawEntry){ -1, isoPlayer.y };
        
        // Add world objects to draw list
        for (int i = 0; i < objCount; i++) {
            Vector2 isoPos = GetIsometricPos(objects[i].position);
            drawList[drawCount++] = (DrawEntry){ i, isoPos.y };
        }
        
        // Sort by Isometric Y (Southern objects drawn last/on top)
        qsort(drawList, drawCount, sizeof(DrawEntry), CompareDrawEntries);

        BeginDrawing();
            ClearBackground((Color){ 210, 170, 100, 255 }); // Desert Sand Base

            BeginMode2D(camera);
                
                // 1. Draw High-Res Tiled Ground
                DrawGround(res.sandTile, camera);

                // 2. Draw Ground Grid Overlay (If debug or preferred)
                if (debugMode) {
                    int step = 100;
                    for (int x = 0; x <= WORLD_SIZE; x += step) {
                        for (int y = 0; y <= WORLD_SIZE; y += step) {
                            Vector2 p = GetIsometricPos((Vector2){ (float)x, (float)y });
                            DrawLineV(p, GetIsometricPos((Vector2){ (float)x + step, (float)y }), (Color){ 0, 0, 0, 40 });
                            DrawLineV(p, GetIsometricPos((Vector2){ (float)x, (float)y + step }), (Color){ 0, 0, 0, 40 });
                        }
                    }
                }

                // 3. Draw Sorted Entities (Objects, Clutter, & Player)
                for (int i = 0; i < drawCount; i++) {
                    if (drawList[i].id == -1) {
                        Player_Draw(&player, isoPlayer, res.apacheSheet);
                    } else {
                        WorldObject* obj = &objects[drawList[i].id];
                        Vector2 isoPos = GetIsometricPos(obj->position);
                        
                        SpriteMeta meta = { 0 };
                        Texture2D tex = res.environmentSheet;
                        Color tint = WHITE;

                        switch (obj->type) {
                            case OBJ_PALM: 
                                meta = META_PALM;
                                break;
                            case OBJ_ROCK:
                                meta = META_ROCK;
                                break;
                            case OBJ_CLUTTER_ROCK:
                                meta = META_ROCK;
                                tint = (Color){ 200, 180, 150, 255 }; // Tinted to match sand better
                                break;
                            case OBJ_HUT:
                                tex = res.structuresSheet;
                                meta = META_HUT;
                                break;
                            case OBJ_HOTEL:
                                tex = res.structuresSheet;
                                meta = META_HOTEL;
                                break;
                            case OBJ_TOWER:
                                tex = res.structuresSheet;
                                meta = META_TOWER;
                                break;
                            case OBJ_FACTORY:
                                tex = res.objectivesSheet;
                                meta = META_FACTORY;
                                break;
                            case OBJ_REFINERY:
                                tex = res.objectivesSheet;
                                meta = META_REFINERY;
                                break;
                            case OBJ_SILO:
                                tex = res.objectivesSheet;
                                meta = META_SILO;
                                break;
                            case OBJ_DOME:
                                tex = res.objectivesSheet;
                                meta = META_DOME;
                                break;
                            case OBJ_RADAR:
                                tex = res.objectivesSheet;
                                meta = META_RADAR;
                                break;
                            case OBJ_SOLDIER:
                                tex = res.peopleSheet;
                                meta = META_SOLDIER_IDLE;
                                break;
                        }
                        
                        Rectangle dest = { isoPos.x, isoPos.y, meta.source.width * obj->scale, meta.source.height * obj->scale };
                        DrawTexturePro(tex, meta.source, dest, (Vector2){ meta.pivot.x * obj->scale, meta.pivot.y * obj->scale }, 0.0f, tint);

                        if (debugMode) {
                            DrawRectangleLinesEx((Rectangle){ isoPos.x - meta.pivot.x * obj->scale, isoPos.y - meta.pivot.y * obj->scale, dest.width, dest.height }, 1, GREEN);
                            DrawCircleV(isoPos, 2, RED);
                            
                            float collR = GetObjectCollisionRadius(obj->type);
                            if (collR > 0) DrawCircleLines(isoPos.x, isoPos.y, collR, YELLOW);
                        }
                    }
                }

                if (debugMode) {
                    DrawCircleLines(isoPlayer.x, isoPlayer.y, PLAYER_COLLISION_RADIUS, YELLOW);
                }

            EndMode2D();

            // --- HIGH-RES HUD / UI ---
            // Top Status Bar
            DrawRectangle(0, 0, SCREEN_WIDTH, 90, (Color){ 10, 15, 20, 230 });
            DrawRectangleLinesEx((Rectangle){ 0, 0, SCREEN_WIDTH, 90 }, 2, DARKGRAY);
            
            // Text Details
            DrawText("OPERATIONAL THEATER: SANDSTORM", 25, 15, 20, RAYWHITE);
            DrawLine(25, 40, 350, 40, SKYBLUE);
            
            // Telemetry
            DrawText(TextFormat("ALTITUDE: %04d FT", (int)player.height), 25, 52, 22, (player.height > 10 ? LIME : RED));
            DrawText(TextFormat("COORD: %04d, %04d", (int)player.position.x, (int)player.position.y), 240, 52, 22, GOLD);
            
            // Secondary Info
            DrawText("FUEL: [|||||||||||||||||||]", 520, 20, 18, LIME);
            DrawText("LOAD: [||||||||||         ]", 520, 45, 18, SKYBLUE);
            DrawText("F1: TOGGLE GRID", 680, 70, 10, DARKGRAY);

            // CRT Scanline Effect (Subtle)
            for (int i = 0; i < SCREEN_HEIGHT; i += 4) {
                DrawLine(0, i, SCREEN_WIDTH, i, (Color){ 0, 0, 0, 15 });
            }

            // Window Border
            DrawRectangleLinesEx((Rectangle){ 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT }, 10, (Color){ 30, 30, 35, 255 });

        EndDrawing();
    }

    Resources_Unload(&res);
    CloseWindow();

    return 0;
}
