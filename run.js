const GRID_SIZE = 64;
const UPDATE_INTERVAL = 66.66; // 15 fps.

const canvas = document.querySelector("canvas");

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
        format: "float32x2",
        offset: 0,
        shaderLocation: 0,
        // shaderLocation has to be between 0 and 15 (arbitrary).
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

// Storage buffer for the cell data.
const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
const cellStateStorage = [
    device.createBuffer({
        label: "Cell State A",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    }),
    device.createBuffer({
        label: "Cell State B",
        size: cellStateArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
];
// Initialize the cell grid.
for (let i = 0; i < cellStateArray.length; ++i) {
    cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
}
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);

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
        @group(0) @binding(1) var<storage> cell_state: array<u32>;

        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput
        {
            let i = f32(input.instance); // Convert instance_index to float.
            // Wrap the instance IDs to grid coordinates.
            let cell = vec2f(i % grid.x, floor(i / grid.x));
            let cell_offset = cell / grid * 2;
            let curr_state = f32(cell_state[input.instance]);

            // grid_pos moves the square onto the grid (instead of keeping the
            // square's origin at its center). Add 1 to the non-grid aligned
            // pos, then divide by the grid size to make the square origin its
            // bottom left corner.
            //
            // After the grid alignment, 1 is subtracted to place the square in
            // the bottom left grid cell of the whole canvas (0, 0). Using a
            // grid-aligned offset, the square can then be moved relative to
            // grid (0, 0).
            //
            // Also, scale the input position by the cell state. (Hides the square
            // when the cell state is 0 or "dead".)
            let grid_pos = ((input.pos * curr_state + 1) / grid) - 1 + cell_offset;

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

// Create the compute shader in a separate module.
const WORKGROUP_SIZE = 8;
const simulationShaderModule = device.createShaderModule({
    label: "Game of Life simulation shader",
    code: `
        // WG_SIZE passed in via JS template literals.
        // Result: Execute compute in groups of
        //         (8, 8, 1).
        // The codelab mentions that work groups of 64
        // are generally good for parallelism across
        // devices. 8 x 8 == 64.
        //
        // global_invocation_id will range from (0,0,0) to
        // (31, 31, 0). One for each cell.
        //
        // Even the compute shader can reuse uniforms.

        @group(0) @binding(0) var<uniform> grid: vec2f;

        // Using two buffers for read/write cell data per
        // simulation step. (Ping-pong method.) Unless you specify,
        // var<storage> will always be read-only.
        @group(0) @binding(1) var<storage> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

        // Convert cell grid vector to the proper cell index in the storage
        // buffers. (Basically, the opposite of the vertex shader cell position
        // lookup calculation.)
        fn cellIndex(cell: vec2u) -> u32 {
            return (cell.y % u32(grid.y)) * u32(grid.x) +
                   (cell.x % u32(grid.x));
        }

        // Look up if cell is active by grid index. Account for
        // wrap-around on edges.
        fn cellActive(x: u32, y: u32) -> u32 {
            return cellStateIn[cellIndex(vec2(x, y))];
        }

        @compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
            // Get active neighbors for the current cell.
            let activeNeighbors = cellActive(cell.x+1, cell.y+1) +
                                  cellActive(cell.x+1, cell.y) +
                                  cellActive(cell.x+1, cell.y-1) +
                                  cellActive(cell.x, cell.y-1) +
                                  cellActive(cell.x-1, cell.y-1) +
                                  cellActive(cell.x-1, cell.y) +
                                  cellActive(cell.x-1, cell.y+1) +
                                  cellActive(cell.x, cell.y+1);

            let i = cellIndex(cell.xy);

            // Switch on neighbor count.
            switch activeNeighbors {
                case 2: { // 2 neighbors, stay alive.
                    cellStateOut[i] = cellStateIn[i];
                }
                case 3: { // 3 neighbors, become or stay alive.
                    cellStateOut[i] = 1;
                }
                default: { // < 2 or > 3 neighbors, dies.
                    cellStateOut[i] = 0;
                }
            }
        }
    `
});

// Create a bind group layout and pipeline layout.
const bindGroupLayout = device.createBindGroupLayout({
    label: "Cell Bind Group Layout",
    entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
        buffer: {} // Grid uniform buffer. (Default, empty braces is fine.)
    }, {
        binding: 1,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
        buffer: {type:"read-only-storage"} // Cell state input buffer.
    }, {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {type: "storage"} // Cell state output buffer.
    }]
});

// Define a bind group for the uniform data.
const bindGroups = [
    device.createBindGroup({
        label: "Cell renderer bind group A",
            // Use the custom bind group layout.
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: {buffer: uniformBuffer}
        }, {
            binding: 1,
            resource: {buffer: cellStateStorage[0]}
        }, {
            binding: 2,
            resource: {buffer: cellStateStorage[1]}
        }],
    }),
    device.createBindGroup({
        label: "Cell renderer bind group B",
        layout: bindGroupLayout,
        entries: [{
            binding: 0,
            resource: {buffer: uniformBuffer}
        }, {
            binding: 1,
            resource: {buffer: cellStateStorage[1]}
        }, {
            binding: 2,
            resource: {buffer: cellStateStorage[0]}
        }],
    })
];

// This is necessary now that the bind groups have a custom layout.
const pipelineLayout = device.createPipelineLayout({
    label: "Cell Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
});

// Now define the render pipeline to describe how the shaders
// are ran.
const cellPipeline = device.createRenderPipeline({
    label: "Cell pipeline",
    layout: pipelineLayout,
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

// Create a separate pipeline for the compute process.
const simulationPipeline = device.createComputePipeline({
    label: "Simulation pipeline",
    layout: pipelineLayout,
    compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
    }
});

let step = 0;
function render() {
    // Create an encoder.
    const encoder = device.createCommandEncoder();

    // Set up the compute pass for the sim.
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(simulationPipeline);
    computePass.setBindGroup(0, bindGroups[step % 2]);
    // Do the work.
    const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
    computePass.dispatchWorkgroups(workgroupCount, workgroupCount);
    computePass.end();

    // Bump the simulation step and begin the render pass. (This
    // ordering keeps the sim ahead of the render, i.e, the render
    // pass will display the latest compute results.)
    step++;

    // Set up to clear the canvas.
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: [0.25, 0.25, 0.25, 1],
            storeOp: "store",
        }]
    });

    // Draw the grid.

    // Use the pipeline created for the triangles.
    pass.setPipeline(cellPipeline);
        // 0 assigns to @group(0) here.
    pass.setBindGroup(0, bindGroups[step % 2]);
    pass.setVertexBuffer(0, vertexBuffer);
    // arg1: 12 floats / 2 coords (xy) per vert == 6
    // arg2: instance the square draw, one per grid cell (width * height).
    pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);

    // End the pass and submit the commands to execute.
    pass.end();
    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);
}

// Start the render loop.
setInterval(render, UPDATE_INTERVAL);
