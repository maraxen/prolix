/**
 * File Loader - Handles loading PDB and trajectory files
 */

import type { TrajectoryData, AtomData, Frame, PDBAtom, MsgPackFrame } from "./types";

export class FileLoader {
    private pdbFile: File | null = null;
    private trajectoryFile: File | null = null;

    setFile(type: "pdb" | "trajectory", file: File): void {
        if (type === "pdb") {
            this.pdbFile = file;
        } else {
            this.trajectoryFile = file;
        }
    }

    hasPdbFile(): boolean {
        return this.pdbFile !== null;
    }

    async load(): Promise<TrajectoryData> {
        if (!this.pdbFile) {
            throw new Error("No PDB file selected");
        }

        // Parse PDB
        const pdbContent = await this.readFileAsText(this.pdbFile);
        const atoms = this.parsePDB(pdbContent);

        // Parse trajectory if provided
        let trajectory: Frame[] | undefined;
        if (this.trajectoryFile) {
            trajectory = await this.parseTrajectory(this.trajectoryFile, atoms.length);
        }

        return {
            atoms,
            trajectory,
            metadata: {
                pdbFile: this.pdbFile.name,
                trajectoryFile: this.trajectoryFile?.name,
                numAtoms: atoms.length,
                numFrames: trajectory?.length ?? 1,
            },
        };
    }

    private readFileAsText(file: File): Promise<string> {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = () => reject(reader.error);
            reader.readAsText(file);
        });
    }

    private readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as ArrayBuffer);
            reader.onerror = () => reject(reader.error);
            reader.readAsArrayBuffer(file);
        });
    }

    /**
     * Parse PDB file and extract C-alpha atoms (or C4' for nucleic acids)
     */
    private parsePDB(content: string): AtomData[] {
        const lines = content.split("\n");
        const atoms: AtomData[] = [];
        const seenResidues = new Set<string>();

        for (const line of lines) {
            // Only use MODEL 1 - stop at first ENDMDL
            if (line.startsWith("ENDMDL")) {
                break;
            }

            if (!line.startsWith("ATOM") && !line.startsWith("HETATM")) {
                continue;
            }

            const atomName = line.substring(12, 16).trim();
            const resName = line.substring(17, 20).trim();
            const chainId = line.substring(21, 22).trim() || "A";
            const resSeq = parseInt(line.substring(22, 26).trim(), 10);

            // Skip water
            if (resName === "HOH" || resName === "WAT") {
                continue;
            }

            // Create unique residue key
            const resKey = `${chainId}:${resSeq}:${resName}`;

            // For proteins, take only CA atoms
            // For nucleic acids, take C4' atoms
            // For ligands, take all heavy atoms
            const isProtein = this.isAminoAcid(resName);
            const isNucleic = this.isNucleicAcid(resName);

            let shouldInclude = false;
            let atomType: "P" | "D" | "R" | "L" = "P";

            if (isProtein && atomName === "CA") {
                shouldInclude = !seenResidues.has(resKey);
                atomType = "P";
            } else if (isNucleic && atomName === "C4'") {
                shouldInclude = !seenResidues.has(resKey);
                atomType = resName.startsWith("D") ? "D" : "R";
            } else if (!isProtein && !isNucleic) {
                // Ligand - include all non-hydrogen atoms
                const element = line.substring(76, 78).trim() || line.substring(12, 14).trim()[0];
                if (element !== "H") {
                    shouldInclude = true;
                    atomType = "L";
                }
            }

            if (shouldInclude) {
                seenResidues.add(resKey);

                const x = parseFloat(line.substring(30, 38).trim());
                const y = parseFloat(line.substring(38, 46).trim());
                const z = parseFloat(line.substring(46, 54).trim());
                const bFactor = parseFloat(line.substring(60, 66).trim()) || 50;

                atoms.push({
                    coord: [x, y, z],
                    chain: chainId,
                    plddt: Math.min(100, Math.max(0, bFactor)), // Use B-factor as pLDDT proxy
                    atomType,
                    resName,
                    resNum: resSeq,
                });
            }
        }

        return atoms;
    }

    private isAminoAcid(resName: string): boolean {
        const aminoAcids = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
            "HIE", "HID", "HIP", "CYX", "ASH", "GLH", // Protonation variants
        ];
        return aminoAcids.includes(resName.toUpperCase());
    }

    private isNucleicAcid(resName: string): boolean {
        const nucleicAcids = [
            "A", "G", "C", "U", "T", "DA", "DG", "DC", "DT",
            "ADE", "GUA", "CYT", "URA", "THY",
        ];
        return nucleicAcids.includes(resName.toUpperCase());
    }

    /**
     * Parse trajectory file (supports JSON and msgpack formats)
     */
    private async parseTrajectory(file: File, numAtoms: number): Promise<Frame[]> {
        const extension = file.name.split(".").pop()?.toLowerCase();

        if (extension === "json") {
            const content = await this.readFileAsText(file);
            return this.parseJSONTrajectory(content, numAtoms);
        } else if (extension === "msgpack" || extension === "array_record") {
            // For msgpack/array_record, we need the msgpack library
            console.warn("msgpack/array_record parsing requires server-side processing. Using JSON fallback.");
            // Try parsing as JSON first
            try {
                const content = await this.readFileAsText(file);
                return this.parseJSONTrajectory(content, numAtoms);
            } catch (e) {
                throw new Error(
                    "Direct browser loading of array_record files is not supported. " +
                    "Use the Python server: python -m prolix.viewer_server trajectory.array_record"
                );
            }
        }

        throw new Error(`Unsupported trajectory format: ${extension}`);
    }

    private parseJSONTrajectory(content: string, numAtoms: number): Frame[] {
        const data = JSON.parse(content);

        // Handle different JSON formats
        if (Array.isArray(data)) {
            // Array of frames
            return data.map((frame: any) => this.parseFrame(frame, numAtoms));
        } else if (data.frames) {
            // Object with frames array
            return data.frames.map((frame: any) => this.parseFrame(frame, numAtoms));
        } else if (data.positions) {
            // Single frame
            return [this.parseFrame(data, numAtoms)];
        }

        throw new Error("Unrecognized trajectory JSON format");
    }

    private parseFrame(frame: any, numAtoms: number): Frame {
        let coords: [number, number, number][];

        if (frame.positions) {
            coords = frame.positions;
        } else if (frame.coords) {
            coords = frame.coords;
        } else {
            throw new Error("Frame missing positions/coords");
        }

        // Validate coordinate count
        if (coords.length !== numAtoms) {
            console.warn(
                `Frame has ${coords.length} positions but PDB has ${numAtoms} atoms. ` +
                `Trajectory may not match structure.`
            );
        }

        return {
            coords: coords as [number, number, number][],
            time_ns: frame.time_ns,
            potential_energy: frame.potential_energy,
            kinetic_energy: frame.kinetic_energy,
        };
    }

    async loadFromUrl(pdbUrl: string, trajectoryUrl: string): Promise<TrajectoryData> {
        // Fetch PDB
        const pdbResponse = await fetch(pdbUrl);
        if (!pdbResponse.ok) {
            throw new Error(`Failed to load PDB from ${pdbUrl}`);
        }
        const pdbContent = await pdbResponse.text();
        const atoms = this.parsePDB(pdbContent);

        console.log(`Parsed ${atoms.length} CA atoms from PDB`);

        // Fetch Trajectory (server now returns CA-only positions)
        let trajectory: Frame[] | undefined;
        const trajResponse = await fetch(trajectoryUrl);
        if (trajResponse.ok) {
            const trajContent = await trajResponse.json();
            console.log(`Trajectory response:`, trajContent);
            // Server returns { frames: [...], num_ca_atoms: N }
            if (trajContent.frames) {
                trajectory = trajContent.frames.map((frame: any) => this.parseFrame(frame, atoms.length));
            }
        } else {
            console.warn(`Failed to load trajectory from ${trajectoryUrl}`);
        }

        return {
            atoms,
            trajectory,
            metadata: {
                pdbFile: "remote.pdb",
                trajectoryFile: trajectory ? "remote.traj" : "none",
                numAtoms: atoms.length,
                numFrames: trajectory?.length ?? 1,
            },
        };
    }
}
