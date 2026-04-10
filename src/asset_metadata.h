#ifndef ASSET_METADATA_H
#define ASSET_METADATA_H

#include <raylib.h>

typedef struct { Rectangle source; Vector2 pivot; } SpriteMeta;

// AI-EXTRACTED APACHE ROTATION (Primary Row)
extern const SpriteMeta APACHE_ROTATION[14];

// MISSION OBJECTIVES (Structures - Objectives.png)
extern const SpriteMeta META_FACTORY;
extern const SpriteMeta META_SILO;
extern const SpriteMeta META_REFINERY;
extern const SpriteMeta META_DOME;
extern const SpriteMeta META_RADAR;

// INFANTRY (Other - People.png)
extern const SpriteMeta META_SOLDIER_IDLE;
extern const SpriteMeta META_SOLDIER_RUN;

// DECORATIONS (Structures - Buildings.png & Natural Elements.png)
extern const SpriteMeta META_HOTEL;
extern const SpriteMeta META_HUT;
extern const SpriteMeta META_TOWER;
extern const SpriteMeta META_ROCK;
extern const SpriteMeta META_PALM;

#endif
