#include "player.h"
#include "asset_metadata.h"
#include <math.h>


// Helper function to convert screen-relative input (normalized vector) to world movement delta.
// This abstracts the transformation: world_x = screen_y + screen_x/2, world_y = screen_y - screen_x/2
Vector2 ScreenToWorldMove(Vector2 screenInput) {
    return (Vector2){
        screenInput.y + (screenInput.x / 2.0f),
        screenInput.y - (screenInput.x / 2.0f)
    };
}

void Player_Init(Player *player) {
    player->position = (Vector2){ 1000, 1000 };
    player->height = 0.0f;
    player->rotation = 0.0f;
    player->speed = 200.0f;
    player->verticalSpeed = 100.0f;
}

void Player_Update(Player *player, float deltaTime) {
    // 1. Horizontal Movement (Screen-relative WASD)
    Vector2 screenInput = { 0, 0 };
    if (IsKeyDown(KEY_W)) screenInput.y -= 1.0f; // Screen UP
    if (IsKeyDown(KEY_S)) screenInput.y += 1.0f; // Screen DOWN
    if (IsKeyDown(KEY_A)) screenInput.x -= 1.0f; // Screen LEFT
    if (IsKeyDown(KEY_D)) screenInput.x += 1.0f; // Screen RIGHT

    if (screenInput.x != 0 || screenInput.y != 0) {
        // Normalize screen input magnitude
        float mag = sqrtf(screenInput.x * screenInput.x + screenInput.y * screenInput.y);
        screenInput.x /= mag;
        screenInput.y /= mag;

        // Calculate world movement delta using the abstracted transformation
        Vector2 worldMove = ScreenToWorldMove(screenInput);

        // Apply movement
        player->position.x += worldMove.x * player->speed * deltaTime;
        player->position.y += worldMove.y * player->speed * deltaTime;

        // Update rotation to face movement direction
        float targetRot = atan2f(worldMove.y, worldMove.x) * RAD2DEG;
        if (targetRot < 0) targetRot += 360.0f;
        player->rotation = targetRot;
    }

    // 2. Elevation Control (Space = UP, Left Ctrl = DOWN)
    if (IsKeyDown(KEY_SPACE)) player->height += player->verticalSpeed * deltaTime;
    if (IsKeyDown(KEY_LEFT_CONTROL)) player->height -= player->verticalSpeed * deltaTime;

    // Clamp height
    if (player->height < 0.0f) player->height = 0.0f;
    if (player->height > 100.0f) player->height = 100.0f;
}

void Player_Draw(const Player *player, Vector2 screenPos, Texture2D sprite) {
    // --- Frame Index Calculation (Heuristic Fix) ---
    // Assuming 12 frames cover 360 degrees evenly (30 degrees per frame).
    // A rotation of 270 degrees (Math North) is assumed to map to frame index 0.
    float angleFromNorth = (270.0f - player->rotation);
    if (angleFromNorth < 0.0f) angleFromNorth += 360.0f;

    // Map this angle difference to an index (0 to 13) by dividing by (360/14).
    int frame = (int)roundf(fmodf(angleFromNorth, 360.0f) / (360.0f / 14.0f)) % 14;

    SpriteMeta meta = APACHE_ROTATION[frame];

    // 1. Shadow Scaling (Depth effect)
    // Scale: fmax(0.4, 1.0 - (Height / 150.0)) ensures shadow fades gradually but never disappears fully.
    float shadowScale = fmaxf(0.4f, 1.0f - (player->height / 150.0f));
    unsigned char alpha = (unsigned char)(100.0f * shadowScale);

    // Draw Shadow
    DrawEllipse(screenPos.x, screenPos.y + 15,
                (int)(40.0f * shadowScale), (int)(12.0f * shadowScale),
                (Color){ 0, 0, 0, alpha });

    // 2. Draw Sprite
    Vector2 renderPos = { screenPos.x, screenPos.y - player->height };
    DrawTexturePro(sprite, meta.source,
                   (Rectangle){ renderPos.x, renderPos.y, meta.source.width, meta.source.height },
                   meta.pivot, 0.0f, WHITE);
}