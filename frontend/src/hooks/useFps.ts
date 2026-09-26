import { useCallback, useEffect, useRef, useState } from "react";

export function useFps(active: boolean): number {
  const [fps, setFps] = useState(0);
  const framesRef = useRef(0);
  const lastTickRef = useRef(performance.now());

  const tick = useCallback(() => {
    framesRef.current += 1;
    const now = performance.now();
    const elapsed = now - lastTickRef.current;
    if (elapsed >= 1000) {
      setFps(Math.round((framesRef.current * 1000) / elapsed));
      framesRef.current = 0;
      lastTickRef.current = now;
    }
  }, []);

  useEffect(() => {
    if (!active) {
      setFps(0);
      return;
    }
    let raf = 0;
    const loop = () => {
      tick();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [active, tick]);

  return fps;
}
