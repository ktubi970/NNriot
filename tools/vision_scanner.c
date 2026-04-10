#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef enum {
    PIVOT_CENTER,
    PIVOT_BOTTOM_CENTER
} PivotStrategy;

void AutoDetectSprites(const char* filename, const char* label, PivotStrategy strategy) {
    Image img = LoadImage(filename);
    if (img.data == NULL) {
        fprintf(stderr, "Failed to load image: %s\n", filename);
        return;
    }

    printf("\n// --- AI Extraction: %s (%s) ---\n", filename, label);
    
    int width = img.width;
    int height = img.height;
    bool* visited = (bool*)calloc(width * height, sizeof(bool));
    
    int entityCount = 0;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Color c = GetImageColor(img, x, y);
            
            // If we find an unvisited non-transparent pixel
            if (c.a > 30 && !visited[y * width + x]) {
                // Ignore Guide Lines
                if ((c.r > 200 && c.g < 50 && c.b > 200) || (c.r < 50 && c.g > 200 && c.b > 200)) {
                    visited[y * width + x] = true;
                    continue;
                }

                int minX = x, maxX = x, minY = y, maxY = y;
                long long sumX = 0, sumY = 0, count = 0;

                // BFS/Flood fill for the entire sprite island
                // Using a simple stack-based flood fill to be more robust than the sweep
                int* stackX = malloc(width * height * sizeof(int));
                int* stackY = malloc(width * height * sizeof(int));
                int stackPtr = 0;

                stackX[stackPtr] = x;
                stackY[stackPtr] = y;
                stackPtr++;
                visited[y * width + x] = true;

                while (stackPtr > 0) {
                    stackPtr--;
                    int cx = stackX[stackPtr];
                    int cy = stackY[stackPtr];

                    if (cx < minX) minX = cx;
                    if (cx > maxX) maxX = cx;
                    if (cy < minY) minY = cy;
                    if (cy > maxY) maxY = cy;
                    sumX += cx; sumY += cy; count++;

                    // Check neighbors
                    int dx[] = {0, 0, 1, -1, 1, 1, -1, -1};
                    int dy[] = {1, -1, 0, 0, 1, -1, 1, -1};
                    for (int i = 0; i < 8; i++) {
                        int nx = cx + dx[i];
                        int ny = cy + dy[i];
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height && !visited[ny * width + nx]) {
                            Color nc = GetImageColor(img, nx, ny);
                            if (nc.a > 30) {
                                // Not a guide line
                                if (!((nc.r > 200 && nc.g < 50 && nc.b > 200) || (nc.r < 50 && nc.g > 200 && nc.b > 200))) {
                                    visited[ny * width + nx] = true;
                                    stackX[stackPtr] = nx;
                                    stackY[stackPtr] = ny;
                                    stackPtr++;
                                }
                            }
                        }
                    }
                }

                if (count > 5) { // Lower threshold for small people
                    float px, py;
                    if (strategy == PIVOT_BOTTOM_CENTER) {
                        px = (maxX - minX + 1) / 2.0f;
                        py = (maxY - minY + 1);
                    } else { // Center of mass
                        px = (float)sumX / count - minX;
                        py = (float)sumY / count - minY;
                    }
                    
                    printf("static const SpriteMeta META_%s_%d = { { %d, %d, %d, %d }, { %.1ff, %.1ff } };\n", 
                           label, entityCount++, minX, minY, (maxX - minX) + 1, (maxY - minY) + 1, px, py);
                }
                free(stackX);
                free(stackY);
            }
        }
    }

    free(visited);
    UnloadImage(img);
}

int main(void) {
    SetTraceLogLevel(LOG_NONE);
    
    // 1. Apache Rotation Frames
    AutoDetectSprites("assets/MS-DOS - Desert Strike_ Return to the Gulf - Vehicles - Apache Helicopter.png", "APACHE", PIVOT_CENTER);
    
    // 2. Large Objectives (Refineries, Silos)
    AutoDetectSprites("assets/MS-DOS - Desert Strike_ Return to the Gulf - Structures - Objectives.png", "OBJ", PIVOT_BOTTOM_CENTER);
    
    // 3. Infantry / People
    AutoDetectSprites("assets/MS-DOS - Desert Strike_ Return to the Gulf - Other - People.png", "INF", PIVOT_BOTTOM_CENTER);
    
    return 0;
}
