/**
 * Controls - UI controls for trajectory playback and viewer options
 */

import type { TrajectoryViewer } from "./viewer";

export class Controls {
    private viewer: TrajectoryViewer;
    private frameCount: number;
    private isPlaying: boolean = false;

    constructor(viewer: TrajectoryViewer, frameCount: number) {
        this.viewer = viewer;
        this.frameCount = frameCount;
        this.setupControls();
    }

    private setupControls(): void {
        // Frame slider
        const slider = document.getElementById("frame-slider") as HTMLInputElement;
        const counter = document.getElementById("frame-counter");

        if (slider) {
            slider.max = String(Math.max(0, this.frameCount - 1));
            slider.value = "0";

            slider.addEventListener("input", () => {
                const frame = parseInt(slider.value, 10);
                this.viewer.setFrame(frame);
                this.updateCounter();
            });
        }

        this.updateCounter();

        // Play button
        const playButton = document.getElementById("play-button") as HTMLButtonElement;
        if (playButton) {
            playButton.addEventListener("click", () => {
                this.isPlaying = this.viewer.togglePlay();
                playButton.textContent = this.isPlaying ? "⏸ Pause" : "▶ Play";
            });
        }

        // Reset button
        const resetButton = document.getElementById("reset-button") as HTMLButtonElement;
        if (resetButton) {
            resetButton.addEventListener("click", () => {
                this.viewer.reset();
                this.isPlaying = false;
                if (playButton) playButton.textContent = "▶ Play";
                if (slider) slider.value = "0";
                this.updateCounter();
            });
        }

        // Color mode
        const colorSelect = document.getElementById("color-mode") as HTMLSelectElement;
        if (colorSelect) {
            colorSelect.addEventListener("change", () => {
                this.viewer.setColorMode(colorSelect.value as "chain" | "rainbow" | "plddt");
            });
        }

        // Shadow toggle
        const shadowCheckbox = document.getElementById("show-shadow") as HTMLInputElement;
        if (shadowCheckbox) {
            shadowCheckbox.addEventListener("change", () => {
                this.viewer.setShadow(shadowCheckbox.checked);
            });
        }

        // Outline toggle
        const outlineCheckbox = document.getElementById("show-outline") as HTMLInputElement;
        if (outlineCheckbox) {
            outlineCheckbox.addEventListener("change", () => {
                this.viewer.setOutline(outlineCheckbox.checked);
            });
        }

        // Listen for frame changes from playback
        window.addEventListener("framechange", ((e: CustomEvent) => {
            const frame = e.detail.frame;
            if (slider) slider.value = String(frame);
            this.updateCounter();
        }) as EventListener);
    }

    private updateCounter(): void {
        const counter = document.getElementById("frame-counter");
        const currentFrame = this.viewer.getFrame();
        if (counter) {
            counter.textContent = `${currentFrame + 1} / ${this.frameCount}`;
        }
    }

    dispose(): void {
        // Clean up event listeners if needed
    }
}
