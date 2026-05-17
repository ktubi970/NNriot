#include <stdio.h>
#include <math.h>
#include "iso_transform.h"
#include "iso_config.h"

static int testsRun    = 0;
static int testsFailed = 0;

#define EXPECT(condition, message) do {                                 \
    testsRun++;                                                         \
    if (!(condition)) {                                                 \
        testsFailed++;                                                  \
        fprintf(stderr, "[FAIL] %s:%d  %s\n", __FILE__, __LINE__, (message)); \
    }                                                                   \
} while (0)

#define ISO_TEST_EPSILON 0.001f

static int Float_NearlyEqual(float a, float b)
{
    return fabsf(a - b) < ISO_TEST_EPSILON;
}

static void Test_OriginMapsToOrigin(void)
{
    Vector2 zero = (Vector2){ 0.0f, 0.0f };
    Vector2 screen = Iso_WorldToScreen(zero);
    EXPECT(Float_NearlyEqual(screen.x, 0.0f), "world(0,0) -> screen.x == 0");
    EXPECT(Float_NearlyEqual(screen.y, 0.0f), "world(0,0) -> screen.y == 0");

    Vector2 world = Iso_ScreenToWorld(zero);
    EXPECT(Float_NearlyEqual(world.x, 0.0f), "screen(0,0) -> world.x == 0");
    EXPECT(Float_NearlyEqual(world.y, 0.0f), "screen(0,0) -> world.y == 0");
}

static void Test_KnownDiamondPoints(void)
{
    /* For a 64x32 diamond tile, (1,0) world should land at half-width,
     * half-height in screen space; (0,1) is mirrored on the X axis. */
    Vector2 east  = Iso_WorldToScreen((Vector2){ 1.0f, 0.0f });
    EXPECT(Float_NearlyEqual(east.x,  (float)ISO_TILE_HALF_WIDTH),  "world(1,0) -> +halfW");
    EXPECT(Float_NearlyEqual(east.y,  (float)ISO_TILE_HALF_HEIGHT), "world(1,0) -> +halfH");

    Vector2 south = Iso_WorldToScreen((Vector2){ 0.0f, 1.0f });
    EXPECT(Float_NearlyEqual(south.x, -(float)ISO_TILE_HALF_WIDTH), "world(0,1) -> -halfW");
    EXPECT(Float_NearlyEqual(south.y,  (float)ISO_TILE_HALF_HEIGHT), "world(0,1) -> +halfH");
}

static void Test_RoundTripWorldToScreen(void)
{
    Vector2 samples[] = {
        (Vector2){ 1.0f,   0.0f   },
        (Vector2){ 0.0f,   1.0f   },
        (Vector2){ 3.5f,  -2.25f  },
        (Vector2){-10.0f,  7.0f   },
        (Vector2){ 42.0f,  17.0f  }
    };
    int count = (int)(sizeof(samples) / sizeof(samples[0]));
    for (int i = 0; i < count; i++) {
        Vector2 screen = Iso_WorldToScreen(samples[i]);
        Vector2 back   = Iso_ScreenToWorld(screen);
        EXPECT(Float_NearlyEqual(back.x, samples[i].x), "round-trip world.x");
        EXPECT(Float_NearlyEqual(back.y, samples[i].y), "round-trip world.y");
    }
}

int main(void)
{
    printf("Desert Strike - iso_transform tests\n");
    printf("------------------------------------\n");

    Test_OriginMapsToOrigin();
    Test_KnownDiamondPoints();
    Test_RoundTripWorldToScreen();

    printf("------------------------------------\n");
    printf("Tests run:    %d\n", testsRun);
    printf("Tests failed: %d\n", testsFailed);

    return (testsFailed == 0) ? 0 : 1;
}
