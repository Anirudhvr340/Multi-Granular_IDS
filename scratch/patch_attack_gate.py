with open('nids/attack_gate.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """        # ---- Weighted fusion of attack probabilities -----------------------
        fused_attack_prob = 0.0
        models_attacking  = 0

        if available_ok:
            total_weight = 0.0
            weighted_sum = 0.0

            for granularity, record in available_ok.items():
                # Base weight from granularity
                base_w = _GRANULARITY_BASE_WEIGHT.get(granularity, 0.25)

                # Dynamic session sparsity attenuation:
                # A 30-flow sequential model cannot form a valid sequence on sparse windows
                if granularity == GRANULARITY_SESSION:
                    n_f = record.supporting_evidence.get("flows_in_window", 30) if record.supporting_evidence else 30
                    if n_f < 5:
                        base_w *= 0.10  # Heavily de-weight isolated 1-4 flows
                    elif n_f < 15:
                        base_w *= 0.50  # De-weight sparse partial windows

                # Confidence modulation: high confidence → higher weight
                # Use sigmoid-like scaling so mid-confidence doesn't dominate
                binary_conf = max(record.attack_probability, 1.0 - record.attack_probability)
                conf_factor = 0.5 + 0.5 * math.tanh((binary_conf - 0.5) * 4)

                w = base_w * conf_factor
                total_weight += w
                weighted_sum += w * record.attack_probability

                if record.attack_probability >= self.threshold:
                    models_attacking += 1

            fused_attack_prob = weighted_sum / total_weight if total_weight > 0 else 0.0

        # ---- Agreement score -----------------------------------------------
        n_avail = len(available_ok)
        model_agreement = models_attacking / n_avail if n_avail > 0 else 0.0

        # ---- Attack family routing ----------------------------------------
        fused_family = self._route_attack_family(available_ok, fused_attack_prob)

        return IntegratedEvidence(
            packet=packet_ev,
            flow=flow_ev,
            time_window=time_ev,
            session=session_ev,
            fused_attack_probability=round(fused_attack_prob, 6),
            fused_attack_family=fused_family,
            model_agreement_score=round(model_agreement, 4),
            models_indicating_attack=models_attacking,
            models_available=n_avail,
            window_id=window_id,
        )"""

replacement = """        # ---- Weighted fusion of attack probabilities with Situational Adaptation ----
        fused_attack_prob = 0.0
        models_attacking  = 0
        raw_weights = {}
        normalized_weights: dict[str, float] = {}

        if available_ok:
            total_weight = 0.0

            for granularity, record in available_ok.items():
                # 1. Base prior weight from granularity
                base_w = _GRANULARITY_BASE_WEIGHT.get(granularity, 0.25)

                # 2. Dynamic session sparsity attenuation:
                # A 30-flow sequential BiLSTM requires sufficient sequence continuity
                if granularity == GRANULARITY_SESSION:
                    n_f = record.supporting_evidence.get("flows_in_window", 30) if record.supporting_evidence else 30
                    if n_f < 5:
                        base_w *= 0.10  # Heavily de-weight isolated 1-4 flows
                    elif n_f < 15:
                        base_w *= 0.50  # De-weight sparse partial windows

                # 3. Dynamic Time-Window temporal buffer maturity:
                # Macro temporal lag/trend features need rolling history to avoid cold-start noise
                elif granularity == GRANULARITY_WINDOW:
                    h_depth = record.supporting_evidence.get("history_depth", 5) if record.supporting_evidence else 5
                    if h_depth < 3:
                        base_w *= 0.40  # De-weight cold-start window models
                    elif h_depth >= 5:
                        base_w *= 1.10  # Mature temporal rolling stats

                # 4. Dynamic Packet 1D-CNN volumetric specialization:
                # Dimensionless packet ratios are primary sensors for high-rate flood anomalies
                elif granularity == GRANULARITY_PACKET:
                    if record.is_attack and record.attack_probability >= 0.80:
                        base_w *= 1.25  # Boost packet sensor on confirmed packet header anomalies

                # 5. Dynamic Confidence modulation:
                # High confidence (close to 0.0 or 1.0) gets higher weight; ambiguous (0.50) is penalized
                binary_conf = max(record.attack_probability, 1.0 - record.attack_probability)
                conf_factor = 0.5 + 0.5 * math.tanh((binary_conf - 0.5) * 4)

                w = base_w * conf_factor
                raw_weights[granularity] = w
                total_weight += w

                if record.attack_probability >= self.threshold:
                    models_attacking += 1

            if total_weight > 0:
                weighted_sum = sum(w * available_ok[g].attack_probability for g, w in raw_weights.items())
                fused_attack_prob = weighted_sum / total_weight
                normalized_weights = {g: round(w / total_weight, 4) for g, w in raw_weights.items()}
            else:
                fused_attack_prob = 0.0
                normalized_weights = {g: 0.25 for g in available_ok}

        # ---- Agreement score -----------------------------------------------
        n_avail = len(available_ok)
        model_agreement = models_attacking / n_avail if n_avail > 0 else 0.0

        # ---- Attack family routing ----------------------------------------
        fused_family = self._route_attack_family(available_ok, fused_attack_prob)

        return IntegratedEvidence(
            packet=packet_ev,
            flow=flow_ev,
            time_window=time_ev,
            session=session_ev,
            fused_attack_probability=round(fused_attack_prob, 6),
            fused_attack_family=fused_family,
            model_agreement_score=round(model_agreement, 4),
            models_indicating_attack=models_attacking,
            models_available=n_avail,
            model_weights=normalized_weights,
            window_id=window_id,
        )"""

assert target in text, 'target not found in attack_gate.py'
text = text.replace(target, replacement, 1)

with open('nids/attack_gate.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Successfully patched AttackGate with Situational Dynamic Weighting & Export!')
