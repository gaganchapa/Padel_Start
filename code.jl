using Flux, Images, FileIO, Plots, CUDA

function preprocess_image(image_path)
    if !isfile(image_path)
        error("Image file not found: $image_path")
    end
    
    try
        img = load(image_path)
        
        if all(img .== first(img))
            @warn "Image appears to be blank or uniform"
        end
        
        img_resized = imresize(img, (224, 224))
        
        if eltype(img_resized) <: RGB
            img_gray = Gray.(img_resized)
        else
            img_gray = img_resized  
        end
        
        img_tensor = Float32.(channelview(img_gray))
        img_tensor = reverse(img_tensor, dims=1)
        
        if ndims(img_tensor) == 2
            return reshape(img_tensor, 224, 224, 1, 1)
        else
            return reshape(img_tensor, size(img_tensor)..., 1)
        end
    catch e
        error("Error loading image: $e")
    end
end

image_path = "generated_image_1.png"
try
    global img_tensor = preprocess_image(image_path)
    println("Image tensor shape: ", size(img_tensor))
    
    if sum(abs.(img_tensor)) < 1e-5
        @warn "Processed image appears to be blank! Check image file."
    end
    
    p = heatmap(img_tensor[:,:,1,1], color=:grays, title="Loaded Image")
    savefig(p, "loaded_image_verification.png")
    println("Saved verification image to 'loaded_image_verification.png'")
catch e
    println("Error in image processing: $e")
    println("Creating test pattern instead...")
    test_pattern = repeat(reshape(Float32[i/224 for i in 1:224], :, 1), 1, 224)
    global img_tensor = reshape(test_pattern, 224, 224, 1, 1)
end

model = Chain(
    Conv((3, 3), 1 => 16, relu),
    MaxPool((2,2)),
    Conv((3, 3), 16 => 32, relu),
    MaxPool((2,2)),
    Conv((3, 3), 32 => 64, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(43264, 128, relu),
    Dense(128, 10),
    softmax
)

println("Model architecture:\n", model)

if CUDA.functional()
    println("Running on GPU")
    img_tensor = img_tensor |> gpu
    model = model |> gpu
else
    println("Running on CPU")
end

output = model(img_tensor)
println("Model output:\n", output)

function extract_feature_maps(model, input_tensor)
    feature_maps = []
    x = input_tensor
    
    if isdefined(model, :layers)
        layers = model.layers
    else
        layers = Flux.children(model)
    end
    
    for (i, layer) in enumerate(layers)
        x = layer(x)
        
        if isa(layer, Conv)
            push!(feature_maps, x |> cpu)
        end
    end
    return feature_maps
end

try
    global feature_maps = extract_feature_maps(model, img_tensor)
    println("Extracted $(length(feature_maps)) feature maps")
catch e
    println("Error extracting feature maps: $e")
    global feature_maps = []
end

function save_feature_maps(feature_maps)
    if isempty(feature_maps)
        println("No feature maps to visualize")
        return
    end
    
    for (i, fmap) in enumerate(feature_maps)
        n_channels = size(fmap, 3)
        feature_plots = []
        
        if n_channels < 1
            println("Warning: Feature map $i has no channels")
            continue
        end
        
        for ch in 1:min(n_channels, 16)
            fmap_img = fmap[:, :, ch, 1]
            
            if all(isnan.(fmap_img)) || all(fmap_img .== first(fmap_img))
                println("Warning: Feature map $i, channel $ch is invalid")
                fmap_img = zeros(Float32, size(fmap_img))
            end
            
            f_min, f_max = minimum(fmap_img), maximum(fmap_img)
            if f_max > f_min
                fmap_normalized = (fmap_img .- f_min) ./ (f_max - f_min)
            else
                fmap_normalized = zeros(Float32, size(fmap_img))
            end
            
            push!(feature_plots, heatmap(
                fmap_normalized, 
                title="Ch $ch", 
                aspect_ratio=:equal, 
                color=:viridis,
                axis=false, 
                ticks=false,
                colorbar=false
            ))
        end

        if isempty(feature_plots)
            println("No valid feature plots for layer $i")
            continue
        end

        n_plots = length(feature_plots)
        grid_rows = max(1, ceil(Int, sqrt(n_plots)))
        grid_cols = max(1, ceil(Int, n_plots / grid_rows))
        
        plot_width = max(200, min(200*grid_cols, 1200))
        plot_height = max(200, min(200*grid_rows, 1000))
        
        try
            current_backend = backend()
            try
                pyplot()
            catch
            end
            
            p_features = plot(
                feature_plots..., 
                layout=(grid_rows, grid_cols), 
                size=(plot_width, plot_height), 
                title="Conv Layer $i",
                margin=10Plots.mm,
                dpi=100
            )
            filename = "conv_layer_$(i)_feature_maps.png"
            savefig(p_features, filename)
            println("Saved feature maps for Conv Layer $i as $filename")
            
            try
                backend(current_backend)
            catch
            end
        catch e
            println("Error plotting feature maps for layer $i: $e")
            try
                for (j, plot_obj) in enumerate(feature_plots)
                    savefig(plot_obj, "conv$(i)_channel$(j).png")
                end
                println("Saved individual channel plots for layer $i")
                
                p_features = plot(
                    feature_plots..., 
                    layout=(1, min(n_plots, 6)), 
                    size=(1200, 200), 
                    title="Feature Maps Conv Layer $i (limited preview)",
                    margin=10Plots.mm
                )
                filename = "conv_layer_$(i)_feature_maps_fallback.png"
                savefig(p_features, filename)
                println("Saved feature maps with fallback layout as $filename")
            catch e2
                println("Fallback layout also failed: $e2")
            end
        end
    end
end

save_feature_maps(feature_maps)

function save_model_architecture(model)
    open("model_architecture.txt", "w") do io
        print(io, model)
    end
    println("Saved model architecture as model_architecture.txt")
end

save_model_architecture(model)

function save_input_image_plots()
    img_data = img_tensor[:,:,1,1]
    
    try
        p1 = heatmap(img_data, title="Input Image (Heatmap)", color=:grays)
        savefig(p1, "input_heatmap.png")
        
        p2 = nothing
        try
            p2 = plot(img_data, title="Input Image (Surface)", st=:surface)
            savefig(p2, "input_surface.png")
        catch e
            println("Could not create surface plot: $e")
        end
        
        p3 = nothing
        try
            p3 = contour(img_data, title="Input Image (Contour)")
            savefig(p3, "input_contour.png")
        catch e
            println("Could not create contour plot: $e")
        end
        
        valid_plots = filter(!isnothing, [p1, p2, p3])
        if !isempty(valid_plots)
            layout_size = (1, length(valid_plots))
            plot_width = 400 * length(valid_plots)
            
            p_combined = plot(
                valid_plots..., 
                layout=layout_size, 
                size=(plot_width, 400),
                margin=10Plots.mm
            )
            savefig(p_combined, "input_image_visualizations.png")
            println("Saved input image visualizations as input_image_visualizations.png")
        end
    catch e
        println("Error saving input image plots: $e")
        try
            p_simple = heatmap(
                img_data, 
                title="Input Image", 
                color=:grays,
                size=(400, 400)
            )
            savefig(p_simple, "input_simple.png")
            println("Saved simple input visualization as input_simple.png")
        catch
            println("Could not save any input visualizations")
        end
    end
end

save_input_image_plots()
