from dataclasses import replace
import numpy as np

class MeasurementBackend:
    def __init__(self, model, config):
        self.model, self.config = model, config

    def measure(self, image, psf, positions, noise):
        c=self.config
        if c.backend=="cpp":
            return self.model.measure_sources(image,psf,positions,noise,c.n_tangent,c.nthreads,c.batch_size)
        # Same native q-stamps and detections for a paired backend comparison.
        from cpu_src.reference import measure
        from cpu_src.api import Measurement
        chunks=[]
        for start in range(0,len(positions),c.batch_size):
            q=self.model.prepare_qstamps(image,psf,positions[start:start+c.batch_size],noise,c.nthreads)
            chunks.append(measure(c.model,q,c.precision,c.n_tangent,c.reference_device,c.batch_size))
        if not chunks:
            return self.model.measure_qstamps(np.empty((0,5,self.model.metadata["stamp_size"],self.model.metadata["stamp_size"])),c.n_tangent,c.nthreads)
        return Measurement(np.concatenate([r.e for r in chunks]),np.concatenate([r.R for r in chunks]),
                           np.concatenate([r.dE_dx for r in chunks]) if c.n_tangent==4 else None,
                           {"backend":"torch_autograd","precision":c.precision},np.floor(positions+.5).astype(np.int64))
