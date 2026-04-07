% PITNNController.m
% =====================================================================
% PITNN DAB Converter — MATLAB System Object for Simulink + PLECS Blockset
% Method 3: Use when running PLECS inside Simulink via the PLECS Blockset.
%
% HOW TO USE
% ───────────
% 1. Add this file to your MATLAB path or working directory.
%
% 2. In Simulink, add a "MATLAB System" block and set the System Object
%    class name to: PITNNController
%
% 3. Wire the block:
%      Inputs  (4): V1, V2, iL, Pref
%      Outputs (3): phi1, phi2, phi3
%
% 4. Connect the PLECS Blockset DAB model outputs (V1, V2, iL) to
%    the PITNNController inputs, and connect phi3 output to the
%    PLECS Phase Shift Modulator input port.
%
% 5. Set PITNN_DIR in the properties block to your project folder.
%
% REQUIREMENTS
% ─────────────
%   MATLAB R2021a or later
%   Simulink
%   PLECS Blockset (for the power converter plant)
%   Python environment with torch + pitnn_inference.py available
%   (MATLAB calls Python via its built-in py.* interface)
%
% Copyright (c) 2026 Chukwuemeka Nzeadibe
% Mississippi State University — All Rights Reserved
% =====================================================================

classdef PITNNController < matlab.System & matlab.system.mixin.Propagates

    % ── User-configurable properties ─────────────────────────────
    properties
        % Full path to the folder containing pitnn_inference.py,
        % pitnn_scripted.pt, pitnn_mu.npy, pitnn_sigma.npy
        PITNN_DIR = 'C:\Users\Nzead\Documents\Research\Power electronics control system\Simulation\pitnn_dab';

        % Sample time in seconds (must match PLECS plant sample time)
        SampleTime = 1/100e3;
    end

    % ── Fixed constants ───────────────────────────────────────────
    properties (Constant)
        PHI12    = pi * 0.95    % 2.9845 rad — fixed inner duty
        PHI_MIN  = 0.05
        PHI3_MAX = 1.50
        V1_NOM   = 800.0
        V2_NOM   = 800.0
        FSW      = 100e3
        SEQ_LEN  = 20
        N_FEAT   = 8

        % Normalisation constants from pitnn_mu.npy / pitnn_sigma.npy
        MU = single([800.02515, 800.20203, 25.38188, 2.99314, ...
                     2.99314,   0.47853,   37593.668, 1.00037])
        SIGMA = single([46.25231, 46.19022, 15.18009, 0.00862, ...
                        0.00862,  0.29205,  18769.152, 0.08148])
    end

    % ── Internal state ────────────────────────────────────────────
    properties (Access = private)
        Buffer       % (20 x 8) single rolling history
        Phi3Prev     % previous phi3 output
        PyCtrl       % Python PITNNInference object
        UsePython    % true if Python interface available
    end

    % ── Simulink sizing callbacks ─────────────────────────────────
    methods (Access = protected)

        function num = getNumInputsImpl(~)
            num = 4;   % V1, V2, iL, Pref
        end

        function num = getNumOutputsImpl(~)
            num = 3;   % phi1, phi2, phi3
        end

        function [sz1,sz2,sz3] = getOutputSizeImpl(~)
            sz1=[1 1]; sz2=[1 1]; sz3=[1 1];
        end

        function [dt1,dt2,dt3] = getOutputDataTypeImpl(~)
            dt1='double'; dt2='double'; dt3='double';
        end

        function [c1,c2,c3] = isOutputComplexImpl(~)
            c1=false; c2=false; c3=false;
        end

        function [f1,f2,f3] = isOutputFixedSizeImpl(~)
            f1=true; f2=true; f3=true;
        end

        function sts = getSampleTimeImpl(obj)
            sts = createSampleTime(obj, 'Type', 'Discrete', ...
                                        'SampleTime', obj.SampleTime);
        end

        % ── Initialisation ────────────────────────────────────────
        function setupImpl(obj)
            obj.Buffer   = zeros(obj.SEQ_LEN, obj.N_FEAT, 'single');
            obj.Phi3Prev = 0.22;

            % Try to connect to Python PITNNInference
            try
                if count(py.sys.path, obj.PITNN_DIR) == 0
                    insert(py.sys.path, int32(0), obj.PITNN_DIR);
                end
                module   = py.importlib.import_module('pitnn_inference');
                model_path  = fullfile(obj.PITNN_DIR, 'pitnn_scripted.pt');
                mu_path     = fullfile(obj.PITNN_DIR, 'pitnn_mu.npy');
                sigma_path  = fullfile(obj.PITNN_DIR, 'pitnn_sigma.npy');
                obj.PyCtrl  = module.PITNNInference(model_path, mu_path, sigma_path);
                obj.UsePython = true;
                fprintf('[PITNN] Python inference active\n');
            catch ME
                warning('[PITNN] Python interface failed: %s\nFalling back to MATLAB normalisation + constant phi3.', ME.message);
                obj.UsePython = false;
            end
        end

        % ── Per-step computation ──────────────────────────────────
        function [phi1, phi2, phi3] = stepImpl(obj, V1, V2, iL, Pref)

            % Build 8-feature vector
            v_ratio  = single(V1 * V2) / single(obj.V1_NOM * obj.V2_NOM);
            feat     = single([V1, V2, iL, obj.PHI12, obj.PHI12, ...
                                obj.Phi3Prev, Pref, v_ratio]);

            % Normalise
            feat_norm = (feat - obj.MU) ./ obj.SIGMA;

            % Update rolling buffer
            obj.Buffer(1:end-1, :) = obj.Buffer(2:end, :);
            obj.Buffer(end, :)     = feat_norm;

            % ── Inference ─────────────────────────────────────────
            if obj.UsePython
                % Call Python PITNNInference via MATLAB py.* interface
                try
                    result = obj.PyCtrl.step(double(V1), double(V2), ...
                                              double(iL),  double(Pref));
                    phi3_out = double(result{3});
                catch
                    phi3_out = double(obj.Phi3Prev);
                end
            else
                % Fallback: ONNX via MATLAB Deep Learning Toolbox
                % Uncomment if you have ONNX support installed:
                % persistent net;
                % if isempty(net)
                %     net = importONNXNetwork(fullfile(obj.PITNN_DIR,'pitnn_model.onnx'));
                % end
                % x = reshape(obj.Buffer, [1, obj.SEQ_LEN, obj.N_FEAT]);
                % out = predict(net, x);
                % phi3_out = double(out(3));
                phi3_out = double(obj.Phi3Prev);
            end

            % Clamp and store
            phi3_out     = max(obj.PHI_MIN, min(obj.PHI3_MAX, phi3_out));
            obj.Phi3Prev = phi3_out;

            % Return outputs
            phi1 = double(obj.PHI12);
            phi2 = double(obj.PHI12);
            phi3 = phi3_out;
        end

        % ── Reset ─────────────────────────────────────────────────
        function resetImpl(obj)
            obj.Buffer   = zeros(obj.SEQ_LEN, obj.N_FEAT, 'single');
            obj.Phi3Prev = 0.22;
            if obj.UsePython && ~isempty(obj.PyCtrl)
                try; obj.PyCtrl.reset(); catch; end
            end
        end

    end % methods

    % ── Public utility methods ────────────────────────────────────
    methods

        function delay_us = phi3ToDelayUs(obj, phi3)
            % Convert phi3 (rad) to gate drive delay (µs)
            delay_us = phi3 / (2 * pi * obj.FSW) * 1e6;
        end

        function duty_pct = phi1ToDutyPct(obj, phi1)
            % Convert phi1 (rad) to duty cycle (%)
            duty_pct = (phi1 / pi) * 100;
        end

    end

end
