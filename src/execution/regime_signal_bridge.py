import numpy as np

class RegimeSignalBridge:
    REGIME_MULTIPLIERS = {
        0: 1.0,   # Trending Bull  — full signal
        1: 0.6,   # Trending Bear  — reduced
        2: 0.3,   # High-Vol Choppy — minimal
        3: 0.8,   # Low-Vol Range  — standard
    }

    def translate(self,
                  hmm_posterior: np.ndarray,   # shape (4,) probabilities
                  lstm_signal: float,           # 0-1 probability of up move
                  garch_position_size: float    # 0.1-2.0
                  ) -> float:
        """
        Returns: forecast value in range [-20, +20]
        pysystemtrade external forecast scale.

        Logic:
        1. dominant_regime = argmax(hmm_posterior)
        2. regime_mult = REGIME_MULTIPLIERS[dominant_regime]
        3. direction = (lstm_signal - 0.5) * 2  # -1 to +1
        4. raw_forecast = direction * regime_mult * garch_position_size * 20
        5. clip to [-20, +20]
        """
        dominant_regime = np.argmax(hmm_posterior)
        regime_mult = self.REGIME_MULTIPLIERS[dominant_regime]
        direction = (lstm_signal - 0.5) * 2

        raw_forecast = direction * regime_mult * garch_position_size * 20
        return float(np.clip(raw_forecast, -20.0, 20.0))

    def get_position_weight(self, hmm_posterior: np.ndarray) -> float:
        # Weighted average of REGIME_MULTIPLIERS by posterior probability
        weight = 0.0
        for rid, mult in self.REGIME_MULTIPLIERS.items():
            weight += hmm_posterior[rid] * mult
        return float(weight)

def external_forecast_adapter(raw_data):
    """
    Adapter for pysystemtrade to ingest the external forecast.
    pysystemtrade expects a function that takes raw data and returns a forecast.
    """
    import os
    import psycopg2

    db_url = os.getenv('TIMESCALE_URL', 'postgresql://localhost/xauusd')
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    try:
        cur.execute("SELECT bridge_forecast FROM signals ORDER BY timestamp DESC LIMIT 1")
        result = cur.fetchone()
        forecast = result[0] if result else 0.0
    except Exception:
        forecast = 0.0
    finally:
        cur.close()
        conn.close()

    return forecast
