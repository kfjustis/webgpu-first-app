const GRID_SIZE = 32;

const canvas = document.querySelector("canvas");

console.log("Starting WebGPU section.");

// Check for WebGPU.
if (!navigator.gpu) {
    throw new Error("WebGPU not supported by this browser.");
} else {
    console.log("WebGPU available!");
}

// Request adapter.
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
} else {
    console.log("GPUAdapter found.");
}

// Request the device.
const device = await adapter.requestDevice();

// Configure the canvas to use a WebGPU context.
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat,
});
console.log("Using canvasFormat: " + canvasFormat);

// Vertices for two triangles; split diagonally from the
// bottom left to the top right.
const vertices = new Float32Array([
    -0.8, -0.8,
     0.8, -0.8,
     0.8,  0.8,

    -0.8, -0.8,
     0.8,  0.8,
    -0.8,  0.8,
]);
// Create a vertex buffer for the traingle data.
const vertexBuffer = device.createBuffer({
    label: "Cell vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
// Actually copy the vertex data into the buffer.
device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);
// Define the vertex layout for the GPU to understand.
const vertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
        format: "float32x2", // (x, y)
        offset: 0,
        shaderLocation: 0, // Position var in .vert shader.
        // shaderLocation has to be between 0 and 15.
    }],
};

// Uniform buffer for grid data.
const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
const uniformBuffer = device.createBuffer({
    label: "Grid Uniforms",
    size: uniformArray.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

// Shader data with grid knowledge.
const cellShaderModule = device.createShaderModule({
    label: "Cell shader",
    code: `
        struct VertexInput {
            @location(0) pos: vec2f,
            @builtin(instance_index) instance: u32,
        };

        struct VertexOutput {
            @builtin(position) pos: vec4f,
            @location(0) cell: vec2f, // Make frag shader aware of cells.
        };

        @group(0) @binding(0) var<uniform> grid: vec2f;

        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput
        {
            let i = f32(input.instance); // Convert instance_index to float.
            // Wrap the instance IDs to grid coordinates.
            let cell = vec2f(i % grid.x, floor(i / grid.x));
            let cell_offset = cell / grid * 2;

            // grid_pos moves the square onto the grid (instead of keeping the
            // square's origin at its center). Add 1 to the non-grid aligned
            // pos, then divide by the grid size to make the square origin its
            // bottom left corner.
            //
            // After the grid alignment, 1 is subtracted to place the square in
            // the bottom left grid cell of the whole canvas (0, 0). Using a
            // grid-aligned offset, the square can then be moved relative to
            // grid (0, 0).
            let grid_pos = ((input.pos + 1) / grid) - 1 + cell_offset;

            var output: VertexOutput;
            output.pos = vec4f(grid_pos, 0, 1);
            output.cell = cell;
            return output;
        }

        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
            let c = input.cell / grid;
            // Make the blue channel brighter instead of keeping
            // dark values.
            return vec4f(c.xy, 1-c.x, 1);
        }
    `
});
// Now define the render pipeline to describe how the shaders
// are ran.
const cellPipeline = device.createRenderPipeline({
    label: "Cell pipeline",
    layout: "auto", // Let WebGPU figure out a decent pipeline layout.
    vertex: {
        module: cellShaderModule,
        entryPoint: "vertexMain",
        buffers: [vertexBufferLayout]
    },
    fragment: {
        module: cellShaderModule,
        entryPoint: "fragmentMain",
        targets: [{format: canvasFormat}]
    }
});

// Define a bind group for the uniform data.
const bindGroup = device.createBindGroup({
    label: "Cell renderer bind group",
    layout: cellPipeline.getBindGroupLayout(0), // @group(0).
    entries: [{
        binding: 0,
        resource: {buffer: uniformBuffer}
    }],
});

// Set up to clear the canvas.
const encoder = device.createCommandEncoder();
const pass = encoder.beginRenderPass({
    colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: [0.04, 0.18, 0.73, 1],
        storeOp: "store",
    }]
});
// Use the pipeline created for the triangles.
pass.setPipeline(cellPipeline);
pass.setVertexBuffer(0, vertexBuffer);
pass.setBindGroup(0, bindGroup); // again, @group(0).
// arg1: 12 floats / 2 coords (xy) per vert == 6
// arg2: instance the square draw, one per grid cell (width * height).
pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);
pass.end();
const commandBuffer = encoder.finish();
device.queue.submit([commandBuffer]);
