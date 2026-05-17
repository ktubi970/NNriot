#include <stdio.h>

static int testsRun    = 0;
static int testsFailed = 0;

#define EXPECT(condition, message) do {                                 \
    testsRun++;                                                         \
    if (!(condition)) {                                                 \
        testsFailed++;                                                  \
        fprintf(stderr, "[FAIL] %s:%d  %s\n", __FILE__, __LINE__, (message)); \
    }                                                                   \
} while (0)

static void Test_BootstrapSmoke(void)
{
    EXPECT(1 == 1, "bootstrap smoke (1==1)");
}

int main(void)
{
    printf("Desert Strike - test runner\n");
    printf("--------------------------------\n");

    Test_BootstrapSmoke();

    printf("--------------------------------\n");
    printf("Tests run:    %d\n", testsRun);
    printf("Tests failed: %d\n", testsFailed);

    return (testsFailed == 0) ? 0 : 1;
}
