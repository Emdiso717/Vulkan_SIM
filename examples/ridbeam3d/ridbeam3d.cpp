// Comparison-only RID executable.  The complete RID implementation remains
// in riddfmb3d; this translation unit selects the beam comparison defaults
// without changing the legacy executable or its configuration.
#define RIDBEAM3D_COMPARISON 1
#include "../riddfmb3d/riddfmb3d.cpp"
