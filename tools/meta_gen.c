#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    int x, y, w, h;
    float px, py;
} Item;

int compareX(const void *a, const void *b) {
    return (((Item*)a)->x - ((Item*)b)->x);
}

int main() {
    FILE *f = fopen("tools/catalog.txt", "r");
    if (!f) return 1;

    Item items[300];
    int count = 0;
    char line[256];

    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "VEHICLE")) {
            Item temp;
            if (sscanf(line, "ID %d [VEHICLE]: Rect{ %d, %d, %d, %d } Pivot{ %f, %f }", 
                       &temp.id, &temp.x, &temp.y, &temp.w, &temp.h, &temp.px, &temp.py) == 7) {
                // Filter: Only Row 1 (Y < 100) and actual Helis (Width > 20)
                if (temp.y < 100 && temp.w > 20) {
                    items[count++] = temp;
                }
            }
        }
    }
    fclose(f);

    // Sort Row 0 by X-coordinate to get sequential North -> East frames
    qsort(items, count, sizeof(Item), compareX);

    printf("#ifndef ASSET_METADATA_H\n#define ASSET_METADATA_H\n\n#include <raylib.h>\n\ntypedef struct { Rectangle source; Vector2 pivot; } SpriteMeta;\n\n");
    printf("// AI-DETECTED APACHE CATALOG (Sorted by Grid-X)\n");
    printf("static const SpriteMeta APACHE_FLIGHT_CATALOG[13] = {\n");

    for (int i = 0; i < count && i < 13; i++) {
        printf("    { { %d, %d, %d, %d }, { %.1ff, %.1ff } }, // Frame %d\n", 
               items[i].x, items[i].y, items[i].w, items[i].h, items[i].px, items[i].py, i);
    }
    printf("};\n\n");
    
    printf("static const SpriteMeta META_HOTEL = { { 367, 16, 124, 128 }, { 62.0f, 120.0f } };\n");
    printf("static const SpriteMeta META_HUT   = { { 15, 384, 102, 52 },  { 51.0f, 48.0f  } };\n");
    printf("static const SpriteMeta META_TOWER = { { 15, 32, 101, 128 },  { 50.0f, 124.0f } };\n");
    printf("static const SpriteMeta META_ROCK  = { { 65, 129, 63, 63 },   { 31.0f, 58.0f  } };\n");
    printf("static const SpriteMeta META_PALM  = { { 321, 1, 63, 95 },    { 31.0f, 92.0f  } };\n\n#endif\n");

    return 0;
}
