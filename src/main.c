#include "raylib.h"
#include "game_config.h"

int main(void)
{
    InitWindow(GAME_WINDOW_WIDTH, GAME_WINDOW_HEIGHT, GAME_WINDOW_TITLE);
    SetTargetFPS(GAME_TARGET_FPS);

    Color backgroundColor = (Color){
        GAME_BG_COLOR_R,
        GAME_BG_COLOR_G,
        GAME_BG_COLOR_B,
        GAME_BG_COLOR_A
    };

    while (!WindowShouldClose())
    {
        BeginDrawing();
            ClearBackground(backgroundColor);
            DrawText(GAME_WINDOW_TITLE,
                     HUD_TITLE_X, HUD_TITLE_Y,
                     HUD_TITLE_FONT_SIZE, RAYWHITE);
            DrawText("Bootstrap OK - Raylib + MinGW",
                     HUD_TITLE_X, HUD_SUBTITLE_Y,
                     HUD_SUBTITLE_FONT_SIZE, LIGHTGRAY);
            DrawFPS(GAME_WINDOW_WIDTH - HUD_FPS_RIGHT_MARGIN, HUD_FPS_Y);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
