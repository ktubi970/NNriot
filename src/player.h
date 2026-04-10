#ifndef PLAYER_H
#define PLAYER_H

#include <raylib.h>

typedef struct {
    Vector2 position; // World Top-down (X, Y)
    float height;     // Elevation (Z)
    float rotation;
    float speed;
    float verticalSpeed;
} Player;

void Player_Init(Player *player);
void Player_Update(Player *player, float deltaTime);
void Player_Draw(const Player *player, Vector2 screenPos, Texture2D sprite);

#endif
