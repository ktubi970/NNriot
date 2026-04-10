#ifndef RESOURCES_H
#define RESOURCES_H

#include <raylib.h>

typedef struct {
    Texture2D apacheSheet;      // Main helicopter animations
    Texture2D structuresSheet;  // Buildings (Huts, Bunkers)
    Texture2D environmentSheet; // Natural elements (Palms, Rocks)
    Texture2D objectivesSheet;  // Tactical objectives (Refineries, Silos)
    Texture2D peopleSheet;      // Infantry and small units
    Texture2D sandTile;         // Ground terrain
} Resources;

bool Resources_Load(Resources *res);
void Resources_Unload(Resources *res);

#endif
