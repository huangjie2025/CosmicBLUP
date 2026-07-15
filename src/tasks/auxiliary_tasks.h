#pragma once
#include "config.h"
#include <string>

namespace cosmic {

class PlinkReader;
class PgenReader;
class BgenReader;

// ── Independent functional modes (QC / SBLUP / Prediction) ──────
// These are standalone analysis modes that don't need the full GS pipeline.

void runQcMode(const Config& cfg);
void runSblupMode(const Config& cfg);
void runPredictionMode(const Config& cfg);

} // namespace cosmic
