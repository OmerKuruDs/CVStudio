"""Long-form documentation for the in-app help panel.

Two dictionaries:

* `FEATURE_DOCS` — entries for the app itself (menu items, panels, concepts).
* `OP_DOCS` — entries for every registered operation.

Each entry is HTML (Qt's QTextBrowser subset — `<h2>`, `<h3>`, `<p>`, `<ul>`,
`<li>`, `<b>`, `<i>`, `<code>`). Content is intentionally beginner-friendly:
what the thing does, when to reach for it, and which other features it
composes well with. Keep entries focused — the on-screen panel is for
orientation, not a textbook.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureTopic:
    key: str
    title: str
    body: str


FEATURE_TOPICS: tuple[FeatureTopic, ...] = (
    FeatureTopic(
        key="getting_started",
        title="Getting started",
        body="""
<h2>Getting started</h2>
<p>cvsandbox is an interactive playground for OpenCV pipelines. The typical
workflow is:</p>
<ul>
  <li><b>1. Load something to look at.</b> Use <code>File &rarr; Open
      Image</code>, <code>Open Camera</code>, or <code>Open Video</code> —
      or the matching buttons in the action bar below the image.</li>
  <li><b>2. Add operations.</b> Click an op in the left catalog. It joins
      the bottom node-graph strip, automatically wired to the previous step.</li>
  <li><b>3. Tune the parameters.</b> Click a node and edit its sliders /
      dropdowns in the right panel. The image and histogram update live.</li>
  <li><b>4. Save the result.</b> The teal <b>Save Image</b> button below the
      image writes the full-resolution output to disk. For live capture,
      <b>Record Video</b> streams each processed frame to an mp4 / avi.</li>
</ul>
<p>Two further power features live alongside: <b>Export Code</b> emits a
self-contained Python function reproducing the pipeline, and <b>Batch
Process Folder</b> applies it to every image in a folder.</p>
""",
    ),
    FeatureTopic(
        key="opening_files",
        title="Opening images, video, and camera",
        body="""
<h2>Opening files</h2>
<h3>Open Image</h3>
<p>Loads a single image file. Supported formats include PNG, JPG, BMP,
TIFF, and WebP — anything OpenCV's <code>imread</code> understands.</p>
<p><b>16-bit and floating-point TIFFs</b> are handled specially: their
dynamic range is rescaled to uint8 instead of clipped, so high-bit-depth
microscopy / scanner / HDR exports display correctly instead of saturating
to white.</p>
<h3>Open Camera</h3>
<p>Starts streaming from the default webcam (device index 0). Each frame
flows through the live pipeline; histograms and timing update in real time.
Use <b>Stop Capture</b> to release the device.</p>
<h3>Open Video</h3>
<p>Plays back a video file (mp4, mov, avi, mkv, webm). The frame rate is
gated by the heaviest step in the pipeline — if the pipeline takes 80 ms,
the display tops out near 12 fps regardless of the source's native rate.
This keeps the worker thread from queueing endless backlog.</p>
""",
    ),
    FeatureTopic(
        key="pipeline_editor",
        title="The pipeline / node graph",
        body="""
<h2>Pipeline editor</h2>
<p>The strip at the bottom of the window is your pipeline rendered as a
node graph. Under the hood it is a true DAG (directed acyclic graph), so
each op can in principle take inputs from multiple upstream nodes and feed
multiple downstream consumers. For most workflows the chain you build by
clicking operations is enough.</p>
<h3>Anatomy of a node</h3>
<ul>
  <li><b>Title</b> — the op name. The coloured strip on the left encodes
      its category.</li>
  <li><b>Ports</b> — circles on the left (inputs) and right (output). The
      first node automatically wires from the Source; multi-input ops have
      one circle per port, stacked vertically.</li>
  <li><b>Green / grey chip</b> (top-right region) toggles the node on or
      off; a disabled node passes its input through untouched.</li>
  <li><b>X chip</b> removes the node from the pipeline.</li>
  <li><b>Timing badge</b> beneath the title shows how long the last preview
      run spent on that step.</li>
</ul>
<h3>Wiring extra connections</h3>
<p>Drag from an output port onto another node's input port to add an edge —
useful for feeding a Blend / Apply Mask / Difference op's second input.
Drag from a connected input port to detach or re-route. The system rejects
edges that would form cycles, duplicate connections, or use unknown ports.</p>
<h3>Repositioning</h3>
<p>Drag a node body to move it anywhere on the canvas. The position
persists through refreshes and round-trips through <b>Save Pipeline</b>.
The Source node is also free to move.</p>
""",
    ),
    FeatureTopic(
        key="image_view",
        title="Image view (zoom, pan, split)",
        body="""
<h2>Image view</h2>
<p>The big central panel where the pipeline output is rendered.</p>
<h3>Zoom and pan</h3>
<ul>
  <li><b>Mouse wheel</b> zooms in / out at the cursor location.</li>
  <li><b>Left-drag</b> pans.</li>
  <li><b>Double-click</b> resets to fit-in-view.</li>
</ul>
<p>The view remembers your zoom across parameter changes as long as the
output shape stays the same — only operations that change image dimensions
(e.g. Resize) reset the fit.</p>
<h3>Before / After split (Ctrl+B)</h3>
<p>Toggle a side-by-side comparison: the original image on the left, the
pipeline output on the right. Heights are normalised; mixed grayscale and
BGR pairs are unified into BGR for a sensible display. Tools that overlay
on a single image (ROI rectangle, paste-destination rectangle) are hidden
in split mode because the composite has different coordinates.</p>
""",
    ),
    FeatureTopic(
        key="histogram",
        title="Histogram panel",
        body="""
<h2>Histogram panel</h2>
<p>The small graph at the bottom-right shows the <b>distribution of pixel
intensities</b> in the current pipeline output. X axis is 0-255; Y axis is
how many pixels hit each value (normalised so the tallest bar reaches the
top).</p>
<h3>How to read it</h3>
<ul>
  <li><b>Pile on the left</b> &rarr; image is dark; shadow detail likely
      hidden.</li>
  <li><b>Pile on the right</b> &rarr; image is bright; highlights may be
      clipped (lost detail).</li>
  <li><b>Narrow central spike</b> &rarr; low contrast (fog / haze).</li>
  <li><b>Wide, even spread</b> &rarr; rich tonal range.</li>
  <li><b>Two sharp bars at 0 and 255 only</b> &rarr; healthy binary output
      (Canny, Otsu, In-Range Mask).</li>
</ul>
<h3>Channel overlays</h3>
<p>Single-channel inputs show one grey curve. BGR inputs draw three
translucent curves (blue, green, red) over the same axis so you can spot
colour imbalances — a sun-tinted photo would show the red peak far to the
right of the blue peak.</p>
<h3>Practical use</h3>
<p>Pick threshold values by looking for valleys in the histogram between
two peaks. When tuning CLAHE, watch the histogram broaden as contrast
amplifies. After Canny, confirm the output really is binary (two spikes,
nothing in between) — anything else means the chain isn't producing a clean
edge map.</p>
""",
    ),
    FeatureTopic(
        key="tools_panel",
        title="Image tools sidebar",
        body="""
<h2>Image tools sidebar</h2>
<p>The narrow column of grouped buttons immediately right of the image
mirrors the View menu. Buttons stay in sync with menu state — toggling one
updates the other.</p>
<h3>View</h3>
<ul>
  <li><b>Before/After split</b> (Ctrl+B) — see the original and the
      pipeline output side by side.</li>
  <li><b>Downscale large previews</b> — resize inputs longer than 1600 px
      for the preview loop. See the "Downscaling preview mode" topic.</li>
</ul>
<h3>ROI</h3>
<ul>
  <li><b>Select ROI</b> (Ctrl+R) — enter draw mode and rubber-band a
      rectangle.</li>
  <li><b>Clear ROI</b> — drop the region; pipeline goes back to operating
      on the full image.</li>
</ul>
<h3>Paste destination</h3>
<ul>
  <li><b>Randomize paste destination</b> (Ctrl+Shift+R) — drop the
      processed ROI crop onto a random spot inside the image.</li>
  <li><b>Clear paste destination</b> — splice the processed crop back at
      the ROI's own position (default).</li>
</ul>
""",
    ),
    FeatureTopic(
        key="roi",
        title="Region of interest (ROI)",
        body="""
<h2>Region of interest</h2>
<p>Run the pipeline only inside a user-drawn rectangle, leaving the rest of
the image untouched. Useful for tuning a chain against one area before
deciding to apply it globally, or for compositing transformations into a
specific patch.</p>
<h3>Drawing</h3>
<p>Activate <b>Select ROI</b> (Ctrl+R) — the cursor becomes a cross. Click
and drag on the image to rubber-band a rectangle; release to commit. A
green dashed outline marks the region. The pipeline immediately switches
to ROI-only mode and updates the preview.</p>
<h3>Cut and paste somewhere else</h3>
<p>Once an ROI is drawn, dragging inside the green rectangle does
<i>not</i> move the source — it sets the <b>paste destination</b>. A cyan
dotted rectangle follows your cursor; releasing it drops the processed
crop at that spot. The original ROI area stays as-is, so you can stamp
the result wherever you like (or run the menu's "Randomize paste
destination" for a quick random placement).</p>
<h3>Channel and shape mismatches</h3>
<p>If the pipeline produces grayscale output while the source is BGR (or
vice-versa), the channel layout is automatically coerced so the splice
succeeds. Pipelines that change image dimensions (e.g. Resize) skip the
splice and leave the source untouched — the preview shows the original
image so it's obvious the ROI couldn't be applied.</p>
""",
    ),
    FeatureTopic(
        key="downscale_preview",
        title="Downscaling preview mode",
        body="""
<h2>Downscaling preview mode</h2>
<p>Live preview redraws on every parameter change. For high-resolution
inputs (4K photographs, drone footage) that becomes painful — a single
Bilateral pass on a 4032 x 3024 image can take seconds.</p>
<p>The View sidebar's <b>Downscale large previews</b> toggle (on by
default) resizes any source whose longest side exceeds <b>1600 pixels</b>
down to 1600 with INTER_AREA before running the pipeline. The original
full-resolution image is preserved internally and used by:</p>
<ul>
  <li><b>Save Processed Image</b> — output is full-res, regardless of the
      preview setting.</li>
  <li><b>Batch Process Folder</b> — processes whatever the input files are
      natively.</li>
  <li><b>Export Code</b> — generates a function that works on any size.</li>
</ul>
<p>Disable the toggle when you need to inspect very fine detail in the
preview. Re-enable it when the pipeline gets sluggish.</p>
""",
    ),
    FeatureTopic(
        key="timing",
        title="Per-operation timing",
        body="""
<h2>Per-operation timing</h2>
<p>Each node in the bottom strip shows how long the last preview run spent
on that step (e.g. <code>12.4 ms</code>). Disabled or never-executed nodes
show no timing.</p>
<p>Use this to spot bottlenecks: if a Gaussian Blur is taking 200 ms while
everything else is under 10 ms, you have a candidate for a smaller kernel
or for moving it after a downscaling step. The timing is measured by the
preview worker thread, so the numbers include only the op's own work — not
Qt repaints or histogram updates.</p>
""",
    ),
    FeatureTopic(
        key="save_image",
        title="Save Processed Image",
        body="""
<h2>Save Processed Image (Ctrl+S)</h2>
<p>Runs the pipeline on the <b>full-resolution</b> source and writes the
result to disk. The output format follows the chosen extension — PNG keeps
exact pixel values, JPG compresses, TIFF preserves precision (though
cvsandbox writes 8-bit content), BMP / WebP are also supported.</p>
<p>This is the right button when you want to commit the version you've
just tuned. It always uses the original loaded image, so even with
<b>Downscale large previews</b> active you still get a full-res output.</p>
<p>ROI, paste destinations, and multi-input wires are all respected —
whatever you see in the preview is what gets saved (just at native
resolution).</p>
""",
    ),
    FeatureTopic(
        key="record_video",
        title="Record Video",
        body="""
<h2>Record Video</h2>
<p>Only meaningful while a camera or video file is active. Pick a path
(.mp4 or .avi) and the app starts writing every processed frame to a
VideoWriter behind the scenes.</p>
<ul>
  <li>FPS is taken from the source (or 30 fps fallback for cameras that
      don't report).</li>
  <li>The codec is chosen from the file extension — mp4v for .mp4, MJPG
      for .avi.</li>
  <li>Frames that change shape mid-recording (a Resize op flipped, for
      example) are coerced back to the initial size so the writer keeps
      a valid stream.</li>
</ul>
<p><b>Stop Recording</b> finalises the file. The recorder also closes
automatically when the capture ends or the app is closed.</p>
""",
    ),
    FeatureTopic(
        key="save_load_pipeline",
        title="Save and load pipelines",
        body="""
<h2>Save and load pipelines</h2>
<p>Pipelines round-trip through <code>.cvpipe.json</code> files
(<b>File &rarr; Save Pipeline As</b>, <b>Open Pipeline</b>). The serialised
format stores:</p>
<ul>
  <li>Every node — its id, op id, parameters, enabled state, and on-canvas
      position.</li>
  <li>Every edge — full port-aware connections, so multi-input wires
      round-trip too.</li>
  <li>ROI rectangle and paste-destination if either is set.</li>
</ul>
<p>Older v1 files (linear chain only) are auto-migrated on load by
replaying their nodes through the chain builder. You don't have to do
anything to read legacy saves.</p>
""",
    ),
    FeatureTopic(
        key="export_code",
        title="Export Code",
        body="""
<h2>Export Code (Ctrl+E)</h2>
<p>Generates a stand-alone Python function that reproduces the pipeline.
The function takes a single argument (a BGR uint8 ndarray) and returns the
processed result.</p>
<p>The generator walks the graph in topological order. Every node gets its
own <code>step_N</code> variable, so branching and multi-input ops resolve
correctly. The output is verified to match the live pipeline byte-for-byte
in the test suite — what you see is what you get when you copy the code
into your own project.</p>
<p>If the pipeline includes a ROI or composite ops, the generated code
also embeds a <code>_coerce_to_match</code> helper so channel-changing
steps splice back into the source cleanly.</p>
""",
    ),
    FeatureTopic(
        key="dataset_gallery",
        title="Dataset tab",
        body="""
<h2>Dataset tab</h2>
<p>The main window has two tabs at the top — <b>Editor</b> (the default,
with image / panels / pipeline) and <b>Dataset</b>. Switching to the
Dataset tab loads a thumbnail grid of every image in a chosen folder;
click any thumbnail and the app flips back to the Editor with that image
loaded as the active source.</p>
<p><b>File &rarr; Open Dataset…</b> (Ctrl+D) or the matching button in
the action bar jumps to the Dataset tab directly and prompts for a folder
on the first visit.</p>
<h3>Typical workflows</h3>
<ul>
  <li><b>Per-image tuning.</b> Pick a folder, click an image, tweak the
      pipeline for that specific case, hit <b>Save Image</b>, then switch
      back to the Dataset tab and pick the next thumbnail. The previously
      active image stays highlighted so you don't lose your place.</li>
  <li><b>Quick A/B browsing.</b> Compare how the same pipeline behaves on
      a representative set of inputs by clicking through them.</li>
  <li><b>Combine with Bulk Export.</b> Tune the pipeline on a few
      representative images, then run <b>Bulk Export Dataset</b> to apply
      the final settings to the whole folder at once.</li>
</ul>
""",
    ),
    FeatureTopic(
        key="batch",
        title="Bulk Export Dataset",
        body="""
<h2>Bulk Export Dataset</h2>
<p><b>File &rarr; Bulk Export Dataset…</b> opens a modal that applies the
current pipeline to every image in a folder, on a worker thread so the UI
stays responsive.</p>
<h3>Options</h3>
<ul>
  <li><b>Input folder</b> — every PNG / JPG / BMP / TIFF / WebP directly
      inside it is processed.</li>
  <li><b>Output folder</b> — created on demand.</li>
  <li><b>Filename suffix</b> — defaults to <code>_processed</code>; the
      output file becomes <code>{stem}{suffix}{ext}</code>.</li>
  <li><b>Overwrite existing files</b> — off by default so accidental
      reruns don't trample previous results.</li>
</ul>
<p>Errors on individual files don't kill the batch; the dialog shows the
first few failures at the end. <b>Cancel</b> stops between files (the
in-progress one is not interrupted mid-write).</p>
""",
    ),
)


OP_DOCS: dict[str, str] = {
    # ------------------------------------------------------------------ Source
    "source.image": """
<h2>Source</h2>
<p><i>source.image · always present, cannot be removed</i></p>
<p>The original image as loaded from disk or streamed from a camera / video.
The Source is the start of every pipeline. Each new op you add auto-wires from
the previous chain node, but the very first op connects from this Source.</p>
<h3>Where it shines</h3>
<ul>
  <li>Wiring its output into a multi-input op's second input (e.g. Blend's
      <code>b</code> port or Apply Mask's <code>mask</code> port) so the op
      can reference the untouched original alongside the processed chain.</li>
  <li>Cross-checking what a pipeline is changing: feed Source plus the
      end of the chain into <b>Difference</b> to see exactly which pixels
      moved.</li>
</ul>
""",
    # --------------------------------------------------------------- Filtering
    "filtering.gaussian_blur": """
<h2>Gaussian Blur</h2>
<p><i>filtering.gaussian_blur · category: Filtering</i></p>
<p>Smooths an image by convolving with a Gaussian kernel — each output pixel
is the weighted average of its neighbours, with weights falling off in the
shape of a bell curve. The result keeps overall brightness and shape but
loses fine detail and high-frequency noise.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Kernel size</b> — odd integer (e.g. 3, 5, 7…). Larger means more
      blur and slower compute. Practical range: 3-31.</li>
  <li><b>Sigma X</b> — Gaussian standard deviation. <code>0</code> lets
      OpenCV derive a sensible value from the kernel size.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Pre-processing before edge detection so noise doesn't create fake
      edges. Try <b>Gaussian Blur → Canny</b>.</li>
  <li>Removing Gaussian (sensor) noise from photos.</li>
  <li>Faking a depth-of-field / soft background look with a large kernel.</li>
</ul>
""",
    "filtering.median_blur": """
<h2>Median Blur</h2>
<p><i>filtering.median_blur · category: Filtering</i></p>
<p>Replaces each pixel with the <b>median</b> of its neighbourhood. Because
the median ignores extreme values, salt-and-pepper noise vanishes while
edges stay crisp.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Kernel size</b> — odd integer; bigger means stronger denoising
      and more loss of texture.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Scanned documents with dust speckles.</li>
  <li>Old photos with isolated white/black pixel noise.</li>
  <li>Drone footage with sensor-dead pixels.</li>
</ul>
""",
    "filtering.bilateral": """
<h2>Bilateral Filter</h2>
<p><i>filtering.bilateral · category: Filtering</i></p>
<p>Edge-preserving smoothing. Like Gaussian Blur, but the weight of each
neighbour also depends on how similar its colour is — so pixels across a
strong edge contribute less. The result: smooth interiors with sharp
boundaries.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Diameter</b> — pixel neighbourhood. 5 is fast, 9 is balanced,
      larger gets slow.</li>
  <li><b>Sigma color</b> — colour distance tolerance. Higher means more
      smoothing across colour edges (less edge preservation).</li>
  <li><b>Sigma space</b> — spatial distance tolerance. Higher means
      farther pixels can still influence each other.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Portrait skin smoothing without softening eyes / hair.</li>
  <li>Cartoon / illustration look: <b>Bilateral → Canny</b> drawn on top
      of the bilateral output.</li>
  <li>Denoising when you must keep object outlines crisp.</li>
</ul>
""",
    "filtering.nl_means": """
<h2>NL-Means Denoise</h2>
<p><i>filtering.nl_means · category: Filtering</i></p>
<p>Non-local means: for each patch, find similar-looking patches elsewhere
in the image and average them together. Very slow but produces the cleanest
result for Gaussian sensor noise.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Strength (h)</b> — filter strength. Higher cleans more but
      smudges fine detail.</li>
  <li><b>Template size</b> — odd, the patch we compare. 7 is typical.</li>
  <li><b>Search size</b> — odd, area scanned for similar patches.
      Larger gets dramatically slower.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>High-ISO photographs.</li>
  <li>Astronomy / microscopy where Gaussian noise dominates.</li>
  <li>Pre-processing before contour detection on noisy frames.</li>
</ul>
<p><i>Tip:</i> turn on <b>Downscale large previews</b> while tuning — this
op is the slowest in the catalog.</p>
""",
    # ---------------------------------------------------------------- Threshold
    "threshold.binary": """
<h2>Binary Threshold</h2>
<p><i>threshold.binary · category: Threshold</i></p>
<p>Splits pixels into two groups: above <code>thresh</code> become
<code>maxval</code> (typically 255), the rest become 0. Input is forced to
grayscale because thresholding is per-pixel-intensity.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Threshold</b> — cutoff value (0-255).</li>
  <li><b>Max value</b> — what bright pixels become.</li>
  <li><b>Invert</b> — flip foreground / background.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Extracting bright objects from a dark background (or vice-versa).</li>
  <li>OCR pre-processing on uniformly lit text.</li>
  <li>Producing masks for <b>Apply Mask</b>.</li>
</ul>
<p>Watch the <b>Histogram</b> panel as you drag the slider — pick a value
that sits in a valley between two peaks.</p>
""",
    "threshold.otsu": """
<h2>Otsu Threshold</h2>
<p><i>threshold.otsu · category: Threshold</i></p>
<p>Picks the binary threshold automatically by analysing the image
histogram — it finds the cutoff that best separates the two dominant
brightness groups (bimodal assumption).</p>
<h3>Where it shines</h3>
<ul>
  <li>Documents and printed pages.</li>
  <li>Backlit objects on a uniform background.</li>
  <li>Any time the histogram has two distinct peaks.</li>
</ul>
<p>If your histogram has a single broad hump or many peaks, reach for
<b>Adaptive Threshold</b> instead.</p>
""",
    "threshold.adaptive": """
<h2>Adaptive Threshold</h2>
<p><i>threshold.adaptive · category: Threshold</i></p>
<p>Computes a separate threshold for each small neighbourhood. Robust to
uneven lighting — the bright side of a page and the shadowed side both get
their own correct cutoff.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Method</b> — Mean (faster) or Gaussian-weighted (smoother).</li>
  <li><b>Block size</b> — odd, ≥3. Neighbourhood used per pixel.</li>
  <li><b>C</b> — constant subtracted from the local mean. Higher
      <code>C</code> = stricter cutoff (less foreground).</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Pages photographed under a desk lamp (one side bright).</li>
  <li>Whiteboards with glare.</li>
  <li>Astrophotography where the sky brightness varies.</li>
</ul>
""",
    # --------------------------------------------------------------- Morphology
    "morphology.erode": """
<h2>Erode</h2>
<p><i>morphology.erode · category: Morphology</i></p>
<p>Shrinks bright regions by removing pixels at their boundaries. Equivalent
to "for each pixel, take the minimum of its neighbourhood". Disconnects
thin bridges and removes small bright noise.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Kernel shape</b> — Rectangle, Ellipse, or Cross.</li>
  <li><b>Kernel size</b> — odd integer; bigger removes more.</li>
  <li><b>Iterations</b> — apply N times in a row.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Cleaning up thresholded masks (kill speckles).</li>
  <li>Separating objects that touch by a thin neck.</li>
</ul>
""",
    "morphology.dilate": """
<h2>Dilate</h2>
<p><i>morphology.dilate · category: Morphology</i></p>
<p>Grows bright regions outward — the opposite of erode. Equivalent to
"for each pixel, take the maximum of its neighbourhood". Closes thin gaps
and connects nearby objects.</p>
<h3>Where it shines</h3>
<ul>
  <li>Thickening detected edges so they're easier to trace.</li>
  <li>Joining fragments of a partially-detected object.</li>
  <li>Building a permissive mask that engulfs the target region.</li>
</ul>
""",
    "morphology.open": """
<h2>Open</h2>
<p><i>morphology.open · category: Morphology</i></p>
<p>Erode then dilate. The first pass removes small bright noise; the second
pass restores the surviving objects to roughly their original size. The net
effect: <b>keep big bright shapes, drop small ones</b>.</p>
<h3>Where it shines</h3>
<ul>
  <li>Removing speckle noise from a thresholded mask without shrinking
      the real objects.</li>
  <li>Cleaning up OCR'd text — punctuation specks vanish, letters survive.</li>
</ul>
""",
    "morphology.close": """
<h2>Close</h2>
<p><i>morphology.close · category: Morphology</i></p>
<p>Dilate then erode. The first pass closes small holes; the second restores
the outline. The net effect: <b>fill small gaps inside bright regions
without growing them</b>.</p>
<h3>Where it shines</h3>
<ul>
  <li>Filling holes inside a thresholded object (broken edges → solid blob).</li>
  <li>Repairing dotted-line annotations.</li>
  <li>Joining nearby fragments of the same letter or shape.</li>
</ul>
""",
    # ------------------------------------------------------------------ Edge
    "edge.canny": """
<h2>Canny</h2>
<p><i>edge.canny · category: Edge</i></p>
<p>The classic multi-stage edge detector: gradient + non-maximum suppression
+ hysteresis. Produces a clean <i>binary</i> edge map.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Threshold 1 / Threshold 2</b> — hysteresis bounds. Pixels above
      T2 are seed edges; pixels above T1 are kept only if connected to a
      seed. Rule of thumb: T2 is roughly 2-3 times T1.</li>
  <li><b>Aperture size</b> — Sobel kernel used internally (3, 5, or 7).</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Lane / line detection in photos and video.</li>
  <li>Pre-processing for contour analysis or Hough transforms.</li>
  <li>Quick "comic book" outline effect.</li>
</ul>
<p><i>Tip:</i> always pair with a small <b>Gaussian Blur</b> upstream so
noise doesn't generate spurious edges.</p>
""",
    "edge.sobel": """
<h2>Sobel</h2>
<p><i>edge.sobel · category: Edge</i></p>
<p>First-order image derivative along x and/or y. Produces a grayscale
gradient magnitude image rather than a binary edge map.</p>
<h3>Parameters</h3>
<ul>
  <li><b>dx / dy</b> — derivative order in each axis. (1, 0) = vertical
      edges, (0, 1) = horizontal edges, (1, 1) = combined.</li>
  <li><b>Kernel size</b> — odd, 1-7. Larger smooths more.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Highlighting only horizontal or vertical structure.</li>
  <li>Feature engineering for classical machine vision pipelines.</li>
  <li>Gradient-magnitude maps when Canny is too aggressive.</li>
</ul>
""",
    "edge.laplacian": """
<h2>Laplacian</h2>
<p><i>edge.laplacian · category: Edge</i></p>
<p>Second-order derivative. Responds to rapid intensity changes regardless
of direction — useful for blob and dot detection but very sensitive to
noise.</p>
<h3>Where it shines</h3>
<ul>
  <li>Detecting small bright/dark blobs (stars, cells, defects).</li>
  <li>Image sharpening: add a scaled Laplacian back to the original.</li>
  <li>Measuring image focus / sharpness (variance of Laplacian).</li>
</ul>
<p>Always denoise upstream (Median or Gaussian) — Laplacian amplifies noise
aggressively.</p>
""",
    # ------------------------------------------------------------------ Color
    "color.to_grayscale": """
<h2>To Grayscale</h2>
<p><i>color.to_grayscale · category: Color</i></p>
<p>Collapses a BGR image into a single intensity channel using the
luminance-weighted standard formula. Already-gray inputs pass through.</p>
<h3>Where it shines</h3>
<ul>
  <li>Pre-processing for any op that needs a single channel
      (Threshold, Canny, Sobel, Find Contours).</li>
  <li>Stylising photos (mood, classical look).</li>
  <li>Reducing data volume when you don't need colour.</li>
</ul>
""",
    "color.to_hsv": """
<h2>To HSV</h2>
<p><i>color.to_hsv · category: Color</i></p>
<p>Converts BGR to <b>Hue / Saturation / Value</b> — a colour space where
colours are addressed by their actual hue (red, green, blue, …) rather than
by mixing primaries. Far easier to reason about for colour-based selection.</p>
<h3>Where it shines</h3>
<ul>
  <li>Right before <b>HSV In-Range Mask</b> to pick a colour range.</li>
  <li>Per-channel analysis: a hue histogram shows the dominant colours.</li>
  <li>Designing colour-based segmentation rules.</li>
</ul>
""",
    "color.invert": """
<h2>Invert</h2>
<p><i>color.invert · category: Color</i></p>
<p><code>255 - pixel</code> on every channel. Light becomes dark; dark
becomes light.</p>
<h3>Where it shines</h3>
<ul>
  <li>Photographic negative effect.</li>
  <li>Flipping a mask's foreground / background.</li>
  <li>Algorithms that assume bright = foreground when your subject is
      actually dark.</li>
</ul>
""",
    "color.channel": """
<h2>Extract Channel</h2>
<p><i>color.channel · category: Color</i></p>
<p>Picks a single channel and returns it as a grayscale image. For BGR
input: 0 = blue, 1 = green, 2 = red. For HSV input: 0 = hue, 1 = saturation,
2 = value.</p>
<h3>Where it shines</h3>
<ul>
  <li>"Show me the red channel only" debugging.</li>
  <li>Isolating saturation to find vivid regions.</li>
  <li>Working with infrared / multispectral imagery where one band carries
      the signal.</li>
</ul>
""",
    "color.clahe": """
<h2>CLAHE</h2>
<p><i>color.clahe · category: Color</i></p>
<p>Contrast Limited Adaptive Histogram Equalization. For each small tile,
spreads pixel intensities across the full 0-255 range, then clips and
recombines with neighbours so over-amplification doesn't create halos. For
colour input it operates on the L channel of LAB so colours stay natural.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Clip limit</b> — caps contrast amplification per tile. Higher =
      more aggressive.</li>
  <li><b>Tile grid</b> — N by N tiles. Smaller = more local, larger =
      more global.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Dim photographs and low-light footage.</li>
  <li>Medical imaging (X-ray, mammography) — the textbook use case.</li>
  <li>Recovering detail in shadowed regions without blowing out the
      highlights.</li>
</ul>
""",
    "color.hsv_in_range": """
<h2>HSV In-Range Mask</h2>
<p><i>color.hsv_in_range · category: Color</i></p>
<p>Returns a binary mask: white where the pixel's HSV falls inside the
given (H, S, V) box, black elsewhere. Input must be BGR — the op converts
to HSV internally.</p>
<h3>Parameters</h3>
<ul>
  <li><b>H min / max</b> — hue range (0-179 in OpenCV).</li>
  <li><b>S min / max</b> — saturation (0-255). Lower S = pale / grey
      colours; higher S = vivid colours.</li>
  <li><b>V min / max</b> — value / brightness.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Detecting a coloured ball, sign, or marker.</li>
  <li>Picking "green" vegetation in drone imagery.</li>
  <li>Filtering out background that has a known colour.</li>
</ul>
<p><i>Note:</i> red wraps around the hue circle (0 and 179). To catch all
reds you may need a second range and combine via <b>Blend</b> with
alpha=0.5 or use upstream ops.</p>
""",
    # -------------------------------------------------------------- Geometric
    "geometric.resize": """
<h2>Resize</h2>
<p><i>geometric.resize · category: Geometric</i></p>
<p>Scales the image. Independent X and Y factors plus an interpolation
choice.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Scale X / Y</b> — multiplier (1.0 = unchanged).</li>
  <li><b>Interpolation</b> — Nearest (sharp / blocky), Linear (general
      purpose), Cubic (slightly smoother upscale), Area (best for
      downscaling).</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Speeding up an expensive pipeline by working at half size.</li>
  <li>Preparing inputs for a fixed-size neural network.</li>
  <li>Producing thumbnails or print-resolution variants.</li>
</ul>
""",
    "geometric.rotate": """
<h2>Rotate</h2>
<p><i>geometric.rotate · category: Geometric</i></p>
<p>Rotates around the image centre by the given angle in degrees. Canvas
size is preserved, so corners that rotate outside the frame are clipped.</p>
<h3>Where it shines</h3>
<ul>
  <li>Correcting tilted scans or photographs.</li>
  <li>Data augmentation when generating training images.</li>
  <li>Compass / orientation overlays.</li>
</ul>
""",
    "geometric.flip": """
<h2>Flip</h2>
<p><i>geometric.flip · category: Geometric</i></p>
<p>Mirrors the image horizontally, vertically, or both.</p>
<h3>Where it shines</h3>
<ul>
  <li>Correcting selfie-camera mirroring.</li>
  <li>Data augmentation.</li>
  <li>Comic-strip or graphic-design layouts where you need a flipped twin.</li>
</ul>
""",
    # --------------------------------------------------------------- Analysis
    "analysis.find_contours": """
<h2>Find Contours</h2>
<p><i>analysis.find_contours · category: Analysis</i></p>
<p>Finds the outline of every connected non-zero region in the input
(treating it as binary), then draws those contours in green on top of the
original image. Pair with a threshold or mask upstream.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Retrieval mode</b> — External (only outer outlines) or All (also
      nested holes).</li>
  <li><b>Min area</b> — discard contours smaller than this many pixels.
      Filters out noise blobs.</li>
  <li><b>Line thickness</b> — pixels; <code>-1</code> fills the contour.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Counting objects after threshold (cells, coins, pills).</li>
  <li>Locating shapes for downstream geometric analysis.</li>
  <li>Visualising what a mask actually contains.</li>
</ul>
<p><i>Typical chain:</i> <b>Gaussian Blur → Threshold → Open → Find
Contours</b>.</p>
""",
    # --------------------------------------------------------------- Composite
    "composite.blend": """
<h2>Blend</h2>
<p><i>composite.blend · category: Composite · multi-input</i></p>
<p>Alpha-blends two images: <code>out = (1 - alpha) · a + alpha · b</code>. Drag a
wire from any node's output to the <code>b</code> port to control the
second input — by default it falls back to the Source image.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Alpha</b> — weight of input <code>b</code> (0 = only
      <code>a</code>, 1 = only <code>b</code>).</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Soft glow: <b>Bilateral / NL-Means → Blend (Source on b, alpha=0.4)</b>
      keeps original sharpness with a denoised wash.</li>
  <li>"Selective sharpening" with alpha<0.5 between original and high-pass.</li>
  <li>Watermarking by blending a logo on top with low alpha.</li>
</ul>
""",
    "composite.apply_mask": """
<h2>Apply Mask</h2>
<p><i>composite.apply_mask · category: Composite · multi-input</i></p>
<p>Keeps the input image's pixels where the mask is non-zero, zeros the
rest. The mask is automatically thresholded and resized to match if needed.
Wire a thresholded image or HSV-range mask into the <code>mask</code>
port.</p>
<h3>Where it shines</h3>
<ul>
  <li>Isolating a coloured object: <b>HSV In-Range Mask → Apply Mask</b>
      (with the original colour image on the <code>image</code> port).</li>
  <li>Background removal once you have a foreground mask.</li>
  <li>Compositing only the interesting part of one image onto a backdrop.</li>
</ul>
""",
    "composite.difference": """
<h2>Difference</h2>
<p><i>composite.difference · category: Composite · multi-input</i></p>
<p>Absolute per-pixel difference between two images: <code>out =
|a - b|</code>. Bright pixels mark where the two inputs disagree.</p>
<h3>Where it shines</h3>
<ul>
  <li>Motion detection: <b>Difference</b> between two consecutive video
      frames lights up only the moving parts.</li>
  <li>Quality assurance — comparing a printed scan against an ideal
      reference.</li>
  <li>Building a high-pass filter: <b>Source → Difference (with Gaussian
      Blur output on b)</b> isolates the high-frequency detail.</li>
</ul>
""",
    # ---------------------------------------------------------------- Features
    "features.harris": """
<h2>Harris Corners</h2>
<p><i>features.harris · category: Features</i></p>
<p>Computes the Harris corner response — a per-pixel score that peaks at
locations where the image varies in multiple directions (i.e. corners, not
edges or flat regions). The output draws every pixel whose response exceeds
<code>threshold × peak</code> in red on top of a BGR copy of the input.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Block size</b> — neighbourhood used to build the structure tensor.
      Small (2-3) for crisp corners, larger to absorb noise.</li>
  <li><b>Sobel aperture</b> — odd 3-7. Bigger derivative kernel = smoother
      gradients but worse localisation.</li>
  <li><b>Harris k</b> — the free parameter in <code>det(M) - k·trace(M)²</code>.
      0.04-0.06 is standard.</li>
  <li><b>Threshold (×peak)</b> — fraction of the strongest response that
      counts as a corner. Lower = more (and noisier) corners.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Calibration patterns (checkerboards) where corners are unambiguous.</li>
  <li>A diagnostic overlay to see which areas of a frame are textured
      enough for tracking.</li>
</ul>
""",
    "features.shi_tomasi": """
<h2>Shi-Tomasi Corners</h2>
<p><i>features.shi_tomasi · category: Features</i></p>
<p>Picks the <i>N</i> highest-quality corners under the Shi-Tomasi criterion
(<code>min(λ₁, λ₂)</code> of the structure tensor) and draws each as a green
circle. Generally produces better tracking points than raw Harris.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Max corners</b> — hard cap on the number returned.</li>
  <li><b>Quality level</b> — fraction of the best corner's score; anything
      below is rejected.</li>
  <li><b>Min distance</b> — minimum pixel spacing between kept corners.</li>
  <li><b>Block size</b> — covariance-window size; same role as in Harris.</li>
  <li><b>Marker radius</b> — visual only.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Picking seed points for optical-flow / KLT tracking.</li>
  <li>Pre-visualising "trackable" areas before running a longer pipeline.</li>
</ul>
""",
    "features.fast": """
<h2>FAST Keypoints</h2>
<p><i>features.fast · category: Features</i></p>
<p>FAST (Features from Accelerated Segment Test) tests a small ring of
pixels around each candidate and accepts it as a corner if a contiguous arc
is uniformly brighter or darker than the centre. Extremely cheap to compute
and rotation-invariant, but produces no descriptors — pair with BRIEF or
ORB if you need matching.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Threshold</b> — intensity difference required for the arc to count.
      Lower = more keypoints (and more clutter).</li>
  <li><b>Non-max suppression</b> — collapses clusters of neighbouring
      keypoints to the single strongest one. Keep on for clean results.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Real-time pipelines where compute budget is tight.</li>
  <li>SLAM / visual odometry pre-processing.</li>
</ul>
""",
    "features.orb": """
<h2>ORB Keypoints</h2>
<p><i>features.orb · category: Features</i></p>
<p>Oriented FAST + Rotated BRIEF — a patent-free alternative to SIFT/SURF.
Detects scale- and rotation-invariant keypoints by running FAST at multiple
pyramid levels. Toggle <b>Draw rich keypoints</b> to render each keypoint's
size and orientation as a circle with a radial line.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Max features</b> — upper bound on retained keypoints.</li>
  <li><b>Scale factor</b> — pyramid step ratio. 1.2 is the OpenCV default;
      smaller keeps more levels.</li>
  <li><b>Pyramid levels</b> — how many octaves are searched.</li>
  <li><b>Draw rich keypoints</b> — visual; show keypoint scale/orientation.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Image stitching / panorama pipelines.</li>
  <li>Object recognition where SIFT's licence is a problem.</li>
</ul>
""",
    "features.hough_lines": """
<h2>Hough Lines</h2>
<p><i>features.hough_lines · category: Features</i></p>
<p>Probabilistic Hough line transform. Internally runs Canny to obtain an
edge map, then accumulates straight-line votes and emits the detected
segments. The output draws each segment in green on a BGR copy of the
input.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Canny low / high</b> — thresholds for the internal edge detector.
      If you already have an edge map upstream, set these wide (e.g. 1 /
      255) so Canny passes everything through.</li>
  <li><b>Accumulator threshold</b> — minimum vote count for a line.</li>
  <li><b>Min line length</b> — discards short segments.</li>
  <li><b>Max line gap</b> — bridges small gaps when joining segments.</li>
  <li><b>Line thickness</b> — visual only.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Lane detection in road footage.</li>
  <li>Finding document borders / scan deskew.</li>
  <li>Detecting horizon, grid lines, ruler markings.</li>
</ul>
""",
    "features.hough_circles": """
<h2>Hough Circles</h2>
<p><i>features.hough_circles · category: Features</i></p>
<p>Detects circles using the Hough gradient method. The image is internally
smoothed, gradient-thresholded, and voted into a (centre-x, centre-y, radius)
accumulator. Detected circles are drawn with green rims and red centre
dots.</p>
<h3>Parameters</h3>
<ul>
  <li><b>dp</b> — inverse ratio of accumulator resolution to image
      resolution. 1.0 = full res (slow, accurate); 2.0 = half res (fast).</li>
  <li><b>Min center distance</b> — minimum spacing between detected
      centres. Too small → duplicate detections.</li>
  <li><b>Canny high threshold</b> — upper Canny threshold used internally.</li>
  <li><b>Accumulator threshold</b> — votes needed to confirm a circle.
      Lower = more false positives.</li>
  <li><b>Min/Max radius</b> — search range. Setting both to 0 lets OpenCV
      pick, but constraining the range is far more reliable.</li>
  <li><b>Line thickness</b> — visual only.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Coin / bottle-cap counting on conveyor lines.</li>
  <li>Eye / iris detection.</li>
  <li>Finding round logos or buttons in screenshots.</li>
</ul>
""",
    # -------------------------------------------------- Filtering (advanced)
    "filtering.unsharp_mask": """
<h2>Unsharp Mask</h2>
<p><i>filtering.unsharp_mask · category: Filtering</i></p>
<p>The classic photographic sharpening technique: blur a copy of the image,
subtract it from the original to isolate the high-frequency detail, then add
that detail back scaled by <i>amount</i>. <i>Threshold</i> protects flat
regions from getting noisier.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Blur radius</b> — odd kernel size for the internal Gaussian copy.
      Smaller = sharper but less halo control.</li>
  <li><b>Amount</b> — strength multiplier. 1.0 = standard, 2-3 = aggressive.</li>
  <li><b>Threshold</b> — only sharpen pixels whose detail magnitude exceeds
      this. 0 sharpens everything (and amplifies noise).</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Recovering perceived sharpness after a denoise or downscale step.</li>
  <li>Compensating for soft lenses or focus.</li>
  <li>Reverse application: low Amount + Gaussian Blur upstream = "frequency
      separation" workflow.</li>
</ul>
""",
    "filtering.custom_kernel": """
<h2>Custom Kernel</h2>
<p><i>filtering.custom_kernel · category: Filtering</i></p>
<p>Convolves the image with a 3×3 kernel chosen from a preset list. Each
preset captures a classic effect; <i>strength</i> scales the whole kernel so
the same preset can be subtle or extreme.</p>
<h3>Presets</h3>
<ul>
  <li><b>Identity</b> — pass-through. Verify wiring.</li>
  <li><b>Sharpen / Strong Sharpen</b> — emphasise differences from the
      neighbourhood.</li>
  <li><b>Edge Enhance</b> — Laplacian-style outline.</li>
  <li><b>Emboss</b> — directional shading, gives a 3D-relief look.</li>
  <li><b>Outline</b> — high-contrast edges over a darkened background.</li>
  <li><b>Box Blur 3×3</b> — uniform average; the simplest smoothing.</li>
</ul>
<p><b>Strength</b> linearly scales the kernel — 0 produces a black image,
1 is the preset's nominal effect, 2-3 over-drives it.</p>
""",
    # ----------------------------------------------------------------- Edge
    "edge.scharr": """
<h2>Scharr</h2>
<p><i>edge.scharr · category: Edge</i></p>
<p>A 3×3 derivative operator very similar to Sobel, but using kernel weights
that better approximate the gradient at small kernel sizes. Prefer Scharr
over <b>Sobel</b> when you have to use a 3×3 kernel and accuracy of the
gradient direction matters.</p>
<h3>Parameters</h3>
<ul>
  <li><b>dx / dy</b> — 0 or 1 each. Exactly one of them must be 1 for a
      single direction. If both are 1 the op falls back to |Gx| + |Gy|
      (gradient magnitude). Both 0 returns the grayscale input untouched.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Optical flow / structure-from-motion pre-processing.</li>
  <li>Anywhere Sobel's accuracy at ksize=3 is borderline.</li>
</ul>
""",
    # ------------------------------------------------------------ Frequency
    "freq.fft_magnitude": """
<h2>FFT Magnitude</h2>
<p><i>freq.fft_magnitude · category: Frequency</i></p>
<p>The Discrete Fourier Transform of the image, rendered as a single-channel
spectrum. Bright spots = strong frequency components at that orientation /
spatial frequency. Use this to <i>see</i> which frequencies dominate a
texture or which directions a pattern repeats in.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Log scale</b> — apply <code>log(1 + |F|)</code> so the dynamic
      range fits in 0-255. Off = nearly all-black with a single DC spike.</li>
  <li><b>Shift center</b> — bring the DC (zero-frequency) component to the
      middle. The standard "spectrum" view.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Diagnosing periodic noise (looks like distinct bright dots).</li>
  <li>Comparing textures.</li>
  <li>Pairing with a <b>Low-Pass / High-Pass Filter</b> to preview which
      frequencies will be kept.</li>
</ul>
""",
    "freq.fft_phase": """
<h2>FFT Phase</h2>
<p><i>freq.fft_phase · category: Frequency</i></p>
<p>The phase angle of each DFT coefficient, rescaled to 0-255. Visually
noisy but it actually carries most of the structural information of the
image — a famous experiment swaps phases between two photos and the
output looks like the phase-donor.</p>
<p><b>Shift center</b> works as in FFT Magnitude.</p>
""",
    "freq.low_pass": """
<h2>Low-Pass Filter</h2>
<p><i>freq.low_pass · category: Frequency</i></p>
<p>Suppresses frequency components further than <i>cutoff radius</i> from
the centre of the spectrum, then inverse-transforms back to an image. The
result is the slow / smooth content of the original.</p>
<h3>Filter shapes</h3>
<ul>
  <li><b>Ideal</b> — hard cutoff. Sharp but produces ringing in the spatial
      domain (Gibbs artefacts).</li>
  <li><b>Gaussian</b> — smooth roll-off. No ringing; most natural for
      photographic images.</li>
  <li><b>Butterworth</b> — controllable roll-off via <i>order</i>; higher
      order behaves more like Ideal, lower like Gaussian.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Comparing against a Gaussian Blur — same effect, different cost
      profile.</li>
  <li>Removing periodic noise (combine with Notch filtering when
      adding more ops).</li>
</ul>
""",
    "freq.high_pass": """
<h2>High-Pass Filter</h2>
<p><i>freq.high_pass · category: Frequency</i></p>
<p>The complement of Low-Pass: keeps only frequencies further than
<i>cutoff radius</i> from the centre. The output is the fast / detailed
content of the original — edges, fine texture, noise.</p>
<p><b>Filter shapes</b> behave identically to Low-Pass.</p>
<h3>Where it shines</h3>
<ul>
  <li>Edge detection in the frequency domain — chain into a threshold.</li>
  <li>Frequency-separation retouching (split detail from base tones).</li>
  <li>Removing slow brightness gradients from scanned documents.</li>
</ul>
""",
    "freq.band_pass": """
<h2>Band-Pass Filter</h2>
<p><i>freq.band_pass · category: Frequency</i></p>
<p>Keeps the annular ring of frequencies between <i>inner radius</i> and
<i>outer radius</i>. Everything slower than inner or faster than outer is
suppressed.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Inner radius / Outer radius</b> — radii in pixels of the
      frequency-space ring. If you accidentally set inner ≥ outer the op
      clamps to a tiny valid band.</li>
  <li><b>Filter shape / order</b> — same options as Low-Pass.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Isolating texture at a specific scale (e.g. fingerprint ridges).</li>
  <li>Selectively boosting mid-frequency detail without amplifying noise.</li>
</ul>
""",
    # ----------------------------------------------------- Morphology (advanced)
    "morphology.gradient": """
<h2>Gradient</h2>
<p><i>morphology.gradient · category: Morphology</i></p>
<p>Dilation minus erosion — produces an outline of bright regions. A cheap,
robust edge detector for binary or near-binary inputs where Canny would be
overkill.</p>
<p>Parameters mirror the basic morphology ops: <b>Kernel shape</b>,
<b>Kernel size</b>, and <b>Iterations</b>.</p>
<h3>Where it shines</h3>
<ul>
  <li>Drawing outlines of segmented shapes for visualisation.</li>
  <li>Pre-processing for shape descriptors (perimeter, contour).</li>
</ul>
""",
    "morphology.tophat": """
<h2>Top-Hat</h2>
<p><i>morphology.tophat · category: Morphology</i></p>
<p>Input minus its morphological opening — isolates bright features smaller
than the structuring element. Subtracting the opening removes everything
that survives an erode-then-dilate cycle, leaving only the small bright
specks.</p>
<h3>Where it shines</h3>
<ul>
  <li>Detecting small bright objects on uneven backgrounds (cells under a
      microscope, defects on a sheet of metal).</li>
  <li>Correcting non-uniform illumination — subtract a Top-Hat from the
      original to flatten the background.</li>
</ul>
""",
    "morphology.blackhat": """
<h2>Black-Hat</h2>
<p><i>morphology.blackhat · category: Morphology</i></p>
<p>Closing minus input — mirror of Top-Hat for dark features. Isolates dark
spots smaller than the structuring element on a brighter background.</p>
<h3>Where it shines</h3>
<ul>
  <li>Reading dark text on uneven backgrounds.</li>
  <li>Finding cracks / pits on a bright surface.</li>
</ul>
""",
    # ----------------------------------------------------- Threshold (advanced)
    "threshold.triangle": """
<h2>Triangle Threshold</h2>
<p><i>threshold.triangle · category: Threshold</i></p>
<p>Automatic threshold via the triangle method: the algorithm draws a line
between the histogram peak and the far end, then picks the threshold at the
point with the maximum perpendicular distance to that line. Works well on
images whose histogram has one dominant lobe and a long tail — the case
where Otsu often picks a poor split.</p>
<p>Parameters: <b>Max value</b> (the bright output), <b>Invert</b>
(swap which side becomes bright).</p>
""",
    "threshold.in_range": """
<h2>In-Range (BGR)</h2>
<p><i>threshold.in_range · category: Threshold</i></p>
<p>Returns a binary mask where every BGR channel falls inside its
[low, high] interval. Pixels that match every channel become 255; the rest
become 0. The op normalises low/high if you accidentally swap them.</p>
<h3>Where it shines</h3>
<ul>
  <li>Quick colour pickers when you already know the BGR bounds.</li>
  <li>Cleaning up rendered/synthetic images where target colours are exact.</li>
</ul>
<p>For chromatic targeting on natural images, prefer the
<b>HSV In-Range Mask</b> from the Color category — it's hue-based and far
less sensitive to lighting.</p>
""",
    # --------------------------------------------------------- Segmentation
    "segmentation.distance_transform": """
<h2>Distance Transform</h2>
<p><i>segmentation.distance_transform · category: Segmentation</i></p>
<p>For each foreground pixel, computes the distance to the nearest
background pixel and normalises the result to 0-255. Enable <b>Color map</b>
to render with the jet palette — peaks (centres of large blobs) glow red,
edges fade to blue.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Distance type</b> — L2 (true Euclidean), L1 (city-block), or
      Chessboard. L2 is the most natural for geometric measurements.</li>
  <li><b>Mask size</b> — 3 is fast, 5 is more accurate, <i>Precise</i> uses
      a full Euclidean computation (slowest).</li>
  <li><b>Color map</b> — jet palette on/off.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Building marker images for <b>Watershed</b>.</li>
  <li>Finding the medial axis / skeleton of shapes.</li>
  <li>Quick "object centre" visualisations.</li>
</ul>
""",
    "segmentation.connected_components": """
<h2>Connected Components</h2>
<p><i>segmentation.connected_components · category: Segmentation</i></p>
<p>Labels each connected white blob in the (auto-binarised) input with a
unique integer, then paints each label in its own colour. Background stays
black. Use <b>Min area</b> to drop noise specks before colouring.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Connectivity</b> — 4-way only counts axial neighbours; 8-way also
      counts diagonals.</li>
  <li><b>Min area</b> — discard labels smaller than this many pixels.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Counting separated objects after a threshold step.</li>
  <li>Visualising how morphology / threshold tuning changes the blob
      decomposition.</li>
</ul>
""",
    "segmentation.watershed": """
<h2>Watershed</h2>
<p><i>segmentation.watershed · category: Segmentation</i></p>
<p>Marker-based segmentation. The op runs the full classic recipe for you:
Otsu threshold → morphological opening → distance transform → connected
components → watershed. Final output: the original image (converted to BGR)
with watershed boundaries drawn in red.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Foreground threshold</b> — fraction of the peak distance value
      treated as <i>sure</i> foreground. Lower picks more area as foreground.</li>
  <li><b>Background dilation</b> — iterations of the dilate that builds
      <i>sure</i> background. Higher = wider unknown band.</li>
  <li><b>Noise kernel</b> — opening kernel size used to clean up specks
      before computing the distance transform.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Separating touching objects (cells, coins, fruit) that a plain
      threshold merges into one blob.</li>
  <li>Producing crisp segmentation boundaries for downstream contour
      detection.</li>
</ul>
""",
    "segmentation.grabcut": """
<h2>GrabCut (rect)</h2>
<p><i>segmentation.grabcut · category: Segmentation</i></p>
<p>Iterative foreground extraction. Initialises with a centred rectangle —
<b>Margin</b> controls how much of the border is treated as background —
then refines the segmentation by alternating Gaussian mixture model
estimation and graph-cut optimisation. The output keeps foreground pixels
and blacks out everything else.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Rect margin (%)</b> — fraction of width/height treated as
      background border at init. 10-20% suits most centred subjects.</li>
  <li><b>Iterations</b> — refinement passes. More is slower; gains beyond
      5 are usually marginal.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>One-shot background removal for product photos with the subject
      roughly centred.</li>
  <li>Generating training data masks with minimal manual effort.</li>
</ul>
""",
    # --------------------------------------------------------------- Arithmetic
    "arithmetic.add": """
<h2>Add</h2>
<p><i>arithmetic.add · category: Arithmetic · multi-input</i></p>
<p>Saturated per-pixel sum: <code>out = clip(a + b, 0, 255)</code>. Wire two
images into <code>a</code> and <code>b</code> — channel layouts and sizes
are auto-matched (a grayscale <code>b</code> is broadcast across BGR, and
size mismatches are resized to <code>a</code>'s shape).</p>
<h3>Where it shines</h3>
<ul>
  <li>Combining two soft masks into a brighter union.</li>
  <li>Boosting brightness uniformly by adding a constant gray plate.</li>
</ul>
<p>For weighted blending use <b>Composite → Blend</b> instead.</p>
""",
    "arithmetic.subtract": """
<h2>Subtract</h2>
<p><i>arithmetic.subtract · category: Arithmetic · multi-input</i></p>
<p>Saturated per-pixel difference: <code>out = clip(a - b, 0, 255)</code>.
Unlike <b>Composite → Difference</b>, the order matters and negative
results are clipped instead of taken absolute. Pair the two ops to compare
signed and absolute differences side by side.</p>
<h3>Where it shines</h3>
<ul>
  <li>Background subtraction when you have a clean reference plate.</li>
  <li>Removing a known illumination pattern from a captured frame.</li>
</ul>
""",
    "arithmetic.multiply": """
<h2>Multiply</h2>
<p><i>arithmetic.multiply · category: Arithmetic · multi-input</i></p>
<p>Element-wise product with a scale factor:
<code>out = clip(a · b · scale, 0, 255)</code>. With a binary mask on
<code>b</code> and <code>scale = 1/255</code> the op behaves like
<b>Apply Mask</b> while still respecting <i>soft</i> mask values.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Scale</b> — output scaling. Defaults to <code>1/255</code> so a
      multiplication by a 0-255 mask stays in range.</li>
</ul>
""",
    "arithmetic.bitwise_and": """
<h2>Bitwise AND</h2>
<p><i>arithmetic.bitwise_and · category: Arithmetic · multi-input</i></p>
<p>Per-pixel bitwise AND. With binary masks this is the intersection — the
output is bright only where <i>both</i> inputs are. With photographic
inputs it produces a stylised, posterised look since every channel gets
ANDed bit-by-bit.</p>
""",
    "arithmetic.bitwise_or": """
<h2>Bitwise OR</h2>
<p><i>arithmetic.bitwise_or · category: Arithmetic · multi-input</i></p>
<p>Per-pixel bitwise OR. The mask-union: bright in either input becomes
bright in the output. Quickest way to combine multiple thresholded
regions.</p>
""",
    "arithmetic.bitwise_xor": """
<h2>Bitwise XOR</h2>
<p><i>arithmetic.bitwise_xor · category: Arithmetic · multi-input</i></p>
<p>Per-pixel bitwise XOR. With binary masks this is the symmetric
difference — bright where <i>exactly one</i> of the inputs is bright.
A handy diagnostic when comparing two segmentation attempts: XOR shows
exactly where they disagree.</p>
""",
    # ----------------------------------------------------------------- Stereo
    "stereo.bm": """
<h2>Stereo BM (disparity)</h2>
<p><i>stereo.bm · category: Stereo · multi-input</i></p>
<p>Classic block-matching disparity. Compares small windows along
corresponding rows of a rectified <i>left</i> / <i>right</i> stereo pair
and reports how far each pixel had to shift to find its best match.
Brighter output = closer object.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Num disparities</b> — the search range. <i>Must be a multiple of
      16; the op rounds your value down.</i> Wider = finds closer objects
      but slower.</li>
  <li><b>Block size</b> — odd, ≥5. Larger windows tolerate noise but blur
      depth discontinuities.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Quick depth previews from a calibrated stereo rig.</li>
  <li>Pipelines where speed matters more than smooth depth (robotics
      obstacle detection).</li>
</ul>
<p>For better quality at the cost of speed, use <b>Stereo SGBM</b>.</p>
""",
    "stereo.sgbm": """
<h2>Stereo SGBM (disparity)</h2>
<p><i>stereo.sgbm · category: Stereo · multi-input</i></p>
<p>Semi-global block matching: minimises a global smoothness energy along
many 1-D paths instead of treating each pixel independently. Slower than
plain BM but produces much cleaner depth maps — the standard choice for
feeding 3D reconstruction.</p>
<h3>Parameters</h3>
<ul>
  <li><b>Num disparities</b> — multiple of 16, same role as BM.</li>
  <li><b>Block size</b> — odd, ≥3. SGBM prefers smaller blocks (5-7).</li>
  <li><b>Min disparity</b> — shift the search range; useful when objects
      are very close (positive) or behind the calibration plane (negative).</li>
  <li><b>Uniqueness ratio</b> — % margin the best match must beat its
      runner-up. Higher = stricter, more "unknown" pixels.</li>
  <li><b>Speckle window</b> — connected-region size below which
      disparities are erased as noise. 0 disables.</li>
</ul>
<h3>Where it shines</h3>
<ul>
  <li>Off-line reconstruction from a calibrated camera pair.</li>
  <li>Wherever the depth map will be lifted into a 3D point cloud — the
      smoother input avoids reconstruction noise.</li>
</ul>
""",
    "ai.vlm_query": """
<h2>VLM Q&amp;A (Ollama)</h2>
<p>Sends the current image plus a free-form prompt to a local
<b>Ollama</b> server running a vision-language model (llava, bakllava,
etc.). The reply is shown in the <b>AI Response</b> panel on the right
side — the image itself is returned unchanged so downstream OpenCV
nodes operate on the original pixels.</p>
<h3>Setup</h3>
<ul>
  <li>Install Ollama from <code>ollama.com</code> and start the
      service (default URL <code>http://localhost:11434</code>).</li>
  <li>Pull a vision model once:
      <code>ollama pull llava</code> (or <code>llava:13b</code>,
      <code>bakllava</code>).</li>
</ul>
<h3>Parameters</h3>
<ul>
  <li><b>Prompt</b> — the question/instruction. Examples:
      <i>"Describe this image in one sentence."</i>,
      <i>"How many people are in the photo?"</i>,
      <i>"What's written on the sign?"</i>.</li>
  <li><b>Model</b> — the Ollama tag of the model to use.</li>
  <li><b>Host</b> — Ollama server URL (override if Ollama runs on
      another machine).</li>
  <li><b>Temperature</b> — 0 for deterministic factual answers,
      higher for more creative replies.</li>
  <li><b>Banner font scale</b> — purely visual; tweaking it re-uses
      the cached reply so the model is not re-invoked.</li>
</ul>
<h3>Notes</h3>
<ul>
  <li><b>Streaming.</b> Replies arrive token-by-token; the banner shows
      a "Thinking…" placeholder, then the partial text with a
      <code>&#9612;</code> cursor as each chunk lands.</li>
  <li><b>Auto-cancel.</b> Editing prompt, model, or temperature while a
      stream is in-flight cancels the old worker — the cache is not
      poisoned and Ollama is not kept busy on an abandoned query. Other
      VLM nodes' streams keep running.</li>
  <li>VLM inference is slow (seconds per frame). The final reply is
      cached per <code>(image, prompt, model, temperature)</code>, so
      editing downstream nodes or the font scale does not re-hit Ollama.</li>
  <li>If Ollama is unreachable the banner shows the error message —
      the pipeline does not crash. Errors are cached too (to prevent
      retry storms during live preview); change the prompt or model to
      retry.</li>
  <li><b>Export Code</b> emits a clearly-marked passthrough for this op
      — the runtime dependency (Ollama + a downloaded VLM model) cannot
      be reproduced in a generated standalone script.</li>
</ul>
""",
    "ai.clip_classify": """
<h2>CLIP Zero-shot Classify (HuggingFace)</h2>
<p>Score the current image against a user-supplied list of text labels
using OpenAI's <b>CLIP</b>. Shows the top-K labels with their
similarity scores in the <b>AI Response</b> panel on the right; the
image itself is returned unchanged so downstream OpenCV ops keep
working.</p>
<h3>Setup</h3>
<ul>
  <li>One-time install of the optional AI extras:
      <code>pip install -e .[ai]</code> (pulls in transformers, torch,
      pillow — about 5 GB on disk).</li>
  <li>The first run downloads the chosen CLIP checkpoint (~600 MB for
      the default <code>openai/clip-vit-base-patch32</code>). The
      banner shows "Loading model…" while this happens.</li>
</ul>
<h3>Parameters</h3>
<ul>
  <li><b>Labels</b> — comma- or newline-separated candidate captions.
      Tip: phrasing as "a photo of a X" often improves accuracy.
      Examples: <i>a photo of a cat, a photo of a dog</i>; or
      <i>hard hat present, no hard hat</i>.</li>
  <li><b>HF model name</b> — any CLIP checkpoint on HuggingFace.
      Default <code>openai/clip-vit-base-patch32</code> is the
      smallest/fastest. Larger variants (e.g.
      <code>openai/clip-vit-large-patch14</code>) cost more memory but
      score more accurately.</li>
  <li><b>Top-K results</b> — how many label/score pairs to print on
      the banner.</li>
  <li><b>Banner font scale</b> — visual only; cached results reused.</li>
</ul>
<h3>Notes</h3>
<ul>
  <li><b>Auto-cancel.</b> Editing labels or the model name while a
      classification is in-flight cancels the old worker.</li>
  <li>Scores are softmax probabilities across the label set — they sum
      to ~1.0. A score around <code>1/N</code> means CLIP could not
      distinguish your labels.</li>
  <li>Like the VLM op, results are cached per
      <code>(image, labels, model)</code>. Re-running with the same
      params is instant.</li>
</ul>
""",
    "ai.owlvit_detect": """
<h2>OWL-ViT Zero-shot Detection (HuggingFace)</h2>
<p>Find objects matching free-form text prompts in the image using
Google's <b>OWL-ViT</b>. Bounding boxes are drawn directly on the
image (one color per prompt — this is spatial data, so it has to live
on the pixels), and a textual summary of every detection — label,
score, and bounding-box coordinates — also appears in the
<b>AI Response</b> panel on the right.</p>
<h3>Setup</h3>
<ul>
  <li>One-time install of the optional AI extras:
      <code>pip install -e .[ai]</code> (~5 GB on disk once torch is
      installed).</li>
  <li>The first run downloads the chosen OWL-ViT checkpoint (~1.5 GB
      for the default). The banner shows "Loading model…" while this
      happens.</li>
</ul>
<h3>Parameters</h3>
<ul>
  <li><b>Prompts</b> — comma- or newline-separated free-form text.
      Phrasing as "a photo of a X" generally works best. Each prompt
      gets its own deterministic color in the rendered output.</li>
  <li><b>HF model name</b> — any OWL-ViT checkpoint on HuggingFace.
      Default <code>google/owlvit-base-patch32</code>; the v2 family
      (<code>google/owlv2-base-patch16-ensemble</code>) scores higher
      at the cost of memory.</li>
  <li><b>Score threshold</b> — drops boxes below this confidence.
      Lower = more boxes (some likely false positives).</li>
  <li><b>Box line thickness</b> — visual only.</li>
  <li><b>Label font scale</b> — visual only; cached detections reused.</li>
</ul>
<h3>Notes</h3>
<ul>
  <li><b>Auto-cancel.</b> Editing prompts, the model name, or the score
      threshold while a run is in-flight cancels the old worker.</li>
  <li>Results are cached per
      <code>(image, prompts, model, threshold)</code>. Tweaking only
      thickness or font scale re-renders instantly from the cached
      boxes.</li>
  <li>Empty result set publishes "No objects matched the prompts" to
      the response panel — try lowering the score threshold or
      rephrasing.</li>
</ul>
""",
    "ai.blip2_caption": """
<h2>BLIP-2 Caption (HuggingFace)</h2>
<p>Generate a free-form description of the image using HuggingFace's
<b>BLIP-2</b>. No prompt or label list required — the caption lands in
the <b>AI Response</b> panel on the right and the image is returned
unchanged.</p>
<h3>Setup</h3>
<ul>
  <li>One-time install of the optional AI extras:
      <code>pip install -e .[ai]</code>.</li>
  <li>First run downloads the chosen BLIP-2 checkpoint. The default
      <code>Salesforce/blip2-opt-2.7b</code> is ~7 GB; loads in fp16
      on GPU to fit consumer cards, fp32 on CPU.</li>
</ul>
<h3>Parameters</h3>
<ul>
  <li><b>HF model name</b> — any BLIP-2-compatible checkpoint.</li>
  <li><b>Max caption length (tokens)</b> — longer captions are more
      detailed but take more time and memory.</li>
</ul>
<h3>Notes</h3>
<ul>
  <li><b>Auto-cancel.</b> Editing the model name or token limit while
      a caption is generating cancels the old worker.</li>
  <li>Results are cached per <code>(image, model, max_tokens)</code>.
      Re-selecting the node, or reloading the same image, hits the
      cache instantly.</li>
  <li>BLIP-2 is CPU-tolerant but slow; on a 12 GB GPU expect
      ~1-2 seconds per caption with the default checkpoint.</li>
</ul>
""",
}


def get_op_doc(spec_id: str) -> str:
    """Return the HTML help blob for `spec_id`, or a fallback notice."""
    doc = OP_DOCS.get(spec_id)
    if doc is None:
        return (
            f"<p>No documentation has been written yet for "
            f"<code>{spec_id}</code>.</p>"
        )
    return doc.strip()


def documented_op_ids() -> tuple[str, ...]:
    return tuple(OP_DOCS.keys())


def get_feature_doc(key: str) -> str:
    """Return the HTML help blob for a feature topic, or a fallback notice."""
    for topic in FEATURE_TOPICS:
        if topic.key == key:
            return topic.body.strip()
    return f"<p>No documentation has been written yet for <code>{key}</code>.</p>"


def feature_topic_keys() -> tuple[str, ...]:
    return tuple(topic.key for topic in FEATURE_TOPICS)
