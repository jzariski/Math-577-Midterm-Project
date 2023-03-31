classdef rotationLayer < nnet.layer.Layer

    properties
        % (Optional) Layer properties.

        % Layer properties go here.
        net
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Layer learnable parameters go here.
    end
    
    methods
        function layer = rotationLayer(name, net)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Rotation Layer for Digits";
            
            % Set rotation net
            layer.net = net;

            % Layer constructor function goes here.
        end
        
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X - Input data
            % Outputs:
            %         Z - Outputs of layer forward function
            
            % Layer forward function for prediction goes here.
            angles = predictAngles(X, layer.net);
            matZ = rotate_digits(extractdata(X), angles);
            Z = dlarray(matZ);

        end

        %function [Z, memory] = forward(layer, X)
            % (Optional) Forward input data through the layer at training
            % time and output the result and a memory value.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X - Input data
            % Outputs:
            %         Z - Outputs of layer forward function
            %         memory      - Memory value for custom backward propagation

            % Layer forward function for training goes here.
            %Z = predict(layer, X);
            %memory = [];
        %end

        %function [dLdX1, …, dLdXn, dLdW1, …, dLdWk] = ...
                %backward(layer, X1, …, Xn, Z1, …, Zm, dLdZ1, …, dLdZm, memory)
            % (Optional) Backward propagate the derivative of the loss  
            % function through the layer.
            %
            % Inputs:
            %         layer             - Layer to backward propagate through
            %         X1, ..., Xn       - Input data
            %         Z1, ..., Zm       - Outputs of layer forward function            
            %         dLdZ1, ..., dLdZm - Gradients propagated from the next layers
            %         memory            - Memory value from forward function
            % Outputs:
            %         dLdX1, ..., dLdXn - Derivatives of the loss with respect to the
            %                             inputs
            %         dLdW1, ..., dLdWk - Derivatives of the loss with respect to each
            %                             learnable parameter
            
            % Layer backward function goes here.
        %end
    end
end