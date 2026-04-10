#include "asset_metadata.h"

// AI-EXTRACTED APACHE ROTATION (Primary Row)
const SpriteMeta APACHE_ROTATION[14] = {
    { { 46, 52, 28, 30 },   { 14.0f, 15.0f } }, // N
    { { 129, 53, 32, 29 },  { 16.0f, 14.5f } }, // NNE
    { { 207, 53, 41, 27 },  { 20.5f, 13.5f } }, // NE
    { { 288, 55, 51, 22 },  { 25.5f, 11.0f } }, // ENE
    { { 372, 55, 58, 19 },  { 29.0f, 9.5f } },  // E
    { { 460, 55, 62, 18 },  { 31.0f, 9.0f } },  // ESE
    { { 549, 51, 60, 22 },  { 30.0f, 11.0f } }, // SE
    { { 636, 48, 60, 25 },  { 30.0f, 12.5f } }, // SSE
    { { 725, 44, 57, 29 },  { 28.5f, 14.5f } }, // S
    { { 819, 41, 48, 32 },  { 24.0f, 16.0f } }, // SSW
    { { 913, 38, 40, 35 },  { 20.0f, 17.5f } }, // SW
    { { 1009, 39, 32, 37 }, { 16.0f, 18.5f } }, // WSW
    { { 1102, 39, 28, 36 }, { 14.0f, 18.0f } }, // W
    { { 1195, 39, 31, 36 }, { 15.5f, 18.0f } }, // WNW
};

// MISSION OBJECTIVES
const SpriteMeta META_FACTORY  = { { 16, 368, 128, 136 }, { 64.0f, 130.0f } };
const SpriteMeta META_SILO     = { { 336, 166, 48, 88 },  { 24.0f, 85.0f } };
const SpriteMeta META_REFINERY = { { 25, 137, 133, 119 }, { 66.5f, 115.0f } };
const SpriteMeta META_DOME     = { { 656, 17, 63, 37 },   { 31.5f, 35.0f } };
const SpriteMeta META_RADAR    = { { 560, 32, 45, 29 },   { 22.5f, 25.0f } };

// INFANTRY
const SpriteMeta META_SOLDIER_IDLE = { { 38, 26, 6, 11 }, { 3.0f, 11.0f } };
const SpriteMeta META_SOLDIER_RUN  = { { 134, 26, 7, 11 }, { 3.5f, 11.0f } };

// DECORATIONS
const SpriteMeta META_HOTEL = { { 367, 16, 124, 128 }, { 62.0f, 120.0f } };
const SpriteMeta META_HUT   = { { 15, 384, 102, 52 },  { 51.0f, 48.0f  } };
const SpriteMeta META_TOWER = { { 15, 32, 101, 128 },  { 50.0f, 124.0f } };
const SpriteMeta META_ROCK  = { { 65, 129, 63, 63 },   { 31.0f, 58.0f  } };
const SpriteMeta META_PALM  = { { 321, 1, 63, 95 },    { 31.0f, 92.0f  } };
