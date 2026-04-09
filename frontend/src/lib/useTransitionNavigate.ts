import { useNavigate } from 'react-router-dom';
import { flushSync } from 'react-dom';

type StartVT = (cb: () => void) => void;

/**
 * useTransitionNavigate
 *
 * View Transitions API를 지원하는 브라우저에서 navigate() 호출을
 * document.startViewTransition() 내부에서 실행하여 페이지 전환 애니메이션을 활성화.
 * 미지원 브라우저(Firefox 등)에서는 일반 navigate()로 폴백.
 */
export function useTransitionNavigate() {
  const navigate = useNavigate();
  return (to: string) => {
    const startVT = (document as Document & { startViewTransition?: StartVT }).startViewTransition;
    if (!startVT) {
      navigate(to);
      return;
    }
    startVT(() => { flushSync(() => { navigate(to); }); });
  };
}
