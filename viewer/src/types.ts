/**
 * Type definitions for the trajectory viewer
 */

export interface AtomData {
    coord: [number, number, number];
    chain: string;
    plddt?: number;
    atomType?: "P" | "D" | "R" | "L"; // Protein, DNA, RNA, Ligand
    resName?: string;
    resNum?: number;
}

export interface Frame {
    coords: [number, number, number][];
    time_ns?: number;
    potential_energy?: number;
    kinetic_energy?: number;
}

export interface TrajectoryData {
    atoms: AtomData[];
    trajectory?: Frame[];
    metadata?: {
        pdbFile?: string;
        trajectoryFile?: string;
        numAtoms?: number;
        numFrames?: number;
    };
}

export interface PDBAtom {
    serial: number;
    name: string;
    resName: string;
    chainId: string;
    resSeq: number;
    x: number;
    y: number;
    z: number;
    element: string;
    bFactor: number;
}

export interface MsgPackFrame {
    positions: number[][];
    velocities?: number[][];
    time_ns?: number;
    potential_energy?: number;
    kinetic_energy?: number;
}
