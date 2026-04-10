#include "resources.h"
#include <stdlib.h>

bool Resources_Load(Resources *res) {
    // 1. Load Apache Helicopter Sheet
    res->apacheSheet = LoadTexture("assets/MS-DOS - Desert Strike_ Return to the Gulf - Vehicles - Apache Helicopter.png");
    
    // 2. Load Buildings & Structures
    res->structuresSheet = LoadTexture("assets/MS-DOS - Desert Strike_ Return to the Gulf - Structures - Buildings.png");
    
    // 3. Load Natural Elements (Palm trees, rocks)
    res->environmentSheet = LoadTexture("assets/MS-DOS - Desert Strike_ Return to the Gulf - Structures - Natural Elements.png");
    
    // 4. Load Tactical Objectives
    res->objectivesSheet = LoadTexture("assets/MS-DOS - Desert Strike_ Return to the Gulf - Structures - Objectives.png");
    
    // 5. Load Infantry
    res->peopleSheet = LoadTexture("assets/MS-DOS - Desert Strike_ Return to the Gulf - Other - People.png");

    // 4. Generate Procedural Ground Terrain (Avoiding loading issues)
    Image sandImg = GenImageColor(512, 512, (Color){ 210, 170, 100, 255 });
    for (int i = 0; i < 5000; i++) {
        int x = rand() % 512;
        int y = rand() % 512;
        unsigned char shade = 180 + rand() % 40;
        ImageDrawPixel(&sandImg, x, y, (Color){ shade, shade - 20, shade - 60, 255 });
    }
    res->sandTile = LoadTextureFromImage(sandImg);
    UnloadImage(sandImg);

    // Success Check
    if (res->apacheSheet.id == 0 || res->structuresSheet.id == 0 || 
        res->environmentSheet.id == 0 || res->sandTile.id == 0 ||
        res->objectivesSheet.id == 0 || res->peopleSheet.id == 0) {
        TraceLog(LOG_ERROR, "CRITICAL: Asset Loading Error (Objectives/People). Check your assets/ folder!");
        return false;
    }
    
    TraceLog(LOG_INFO, "Professional Assets Loaded Successfully");
    return true;
}

void Resources_Unload(Resources *res) {
    UnloadTexture(res->apacheSheet);
    UnloadTexture(res->structuresSheet);
    UnloadTexture(res->environmentSheet);
    UnloadTexture(res->objectivesSheet);
    UnloadTexture(res->peopleSheet);
    UnloadTexture(res->sandTile);
    TraceLog(LOG_INFO, "Professional Assets Unloaded Successfully");
}
