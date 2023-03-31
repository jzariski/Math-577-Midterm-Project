classdef preluLayer < nnet.layer.Layer
% Example custom PReLU layer. Modified from
% the code on MathWorks site, in the following way:
%
%   - It supports a single learnable parameter Alpha
%   - It implements methods 'forward' and 'backward'
%     in addition to 'predict'; the newest 2019b code omits
%     these methods; it appears that MATLAB moved to
%     automatic differentiation in the last edition, but
%     we will not rely upon this capability, thus have
%     to differentiate by hand.

    properties (Learnable)
        % Layer learnable parameters
        
        % Scaling coefficient
        Alpha
    end
    
    methods
        function layer = preluLayer(name) 
        % layer = preluLayer(numChannels, name) creates a PReLU layer
        % with numChannels channels and specifies the layer name.

        % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "PReLU with 1 channel";
            
            % Initialize scaling coefficient.
            layer.Alpha = rand([1 1]);
        end
        
        function Z = predict(layer, X)
        % Z = predict(layer, X) forwards the input data X through the
        % layer and outputs the result Z.
            % For multi-observation input, the layer expects an array of observations of
            % size h-by-w-by-c-by-N, where h, w, and c are the height, width,
            % and number of channels, respectively, and N is the number of
            % observations.
            Z = max(0, X) + layer.Alpha .* min(0, X);
        end

        function [Z, memory] = forward(layer, X)
        % Z = predict(layer, X) forwards the input data X through the
        % layer and outputs the result Z.
            Z = predict(layer, X);
            memory = [];
        end
        
        function [dLdX,dLdAlpha] = backward(layer, X, Z, dLdZ, memory)
        % For multi-observation input, the layer expects an array of observations of
        % size h-by-w-by-c-by-N, where h, w, and c are the height, width,
        % and number of channels, respectively, and N is the number of
        % observations.
            dZdX = heaviside(X) + layer.Alpha .* heaviside(-X);
            % NOTE: The Hadamard product is a replacement for
            %       the following pseudo-code
            % dLdX = zeros(size(X),class(X));
            % for n=1:size(X,2)
            %     % for j = 1:size(X,1)
            %     %     dLdX(j,n) =   dLdZ(j,n) * dZdX(j,n);
            %     % end
            % end
            dLdX = dLdZ .* dZdX;

            dZdAlpha = min(0,X);
            %
            % NOTE: The Hadamard product followed by contraction is a replacement for
            %       the following pseudo-code:
            %
            % dLdAlpha = 0;
            % for n=1:size(X,2)            
            %     for j = 1:size(X,1)
            %         dLdAlpha = dLdAlpha + dLdZ(j,n) * dZdAlpha(j,n);
            %     end
            % end
            dLdAlpha = sum(dLdZ .* dZdAlpha,'all');
        end

    end
end