import React, { useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

interface AccessibleDialogProps {
  isOpen: boolean;
  onClose: () => void;
  labelledBy: string;
  describedBy?: string;
  initialFocusRef?: React.RefObject<HTMLElement>;
  variant?: 'modal' | 'drawer';
  className?: string;
  children: React.ReactNode;
}

const openDialogs: symbol[] = [];
let savedBodyOverflow = '';
const backgroundState = new Map<HTMLElement, { inert: boolean; ariaHidden: string | null }>();

const dialogStyles = `
.nn-dialog-overlay { position: fixed; inset: 0; z-index: 100; display: flex; align-items: center; justify-content: center; padding: 1rem; background: rgba(2,6,12,.72); backdrop-filter: blur(8px); opacity: 0; transition: opacity 180ms ease; }
.nn-dialog-overlay[data-open="true"] { opacity: 1; }
.nn-dialog-surface { position: relative; display: flex; flex-direction: column; width: 100%; max-height: calc(100dvh - 2rem); overflow: hidden; background: var(--surface, #191b20); border: 1px solid var(--border, #343840); border-radius: 1rem; box-shadow: 0 24px 80px rgba(0,0,0,.45); color: var(--text, #eef2f6); transform: translateY(12px); transition: transform 180ms ease; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.nn-dialog-overlay[data-open="true"] .nn-dialog-surface { transform: translateY(0); }
.nn-dialog-overlay[data-variant="drawer"] { justify-content: flex-end; padding: 0; }
.nn-dialog-overlay[data-variant="drawer"] .nn-dialog-surface { height: 100dvh; max-height: 100dvh; max-width: 38rem; border-radius: 1rem 0 0 1rem; transform: translateX(24px); }
.nn-dialog-overlay[data-variant="drawer"][data-open="true"] .nn-dialog-surface { transform: translateX(0); }
@media (max-width: 600px) { .nn-dialog-overlay { padding: .5rem; } .nn-dialog-surface { max-height: calc(100dvh - 1rem); } .nn-dialog-overlay[data-variant="drawer"] .nn-dialog-surface { border-radius: 0; } }
@media (prefers-reduced-motion: reduce) { .nn-dialog-overlay, .nn-dialog-surface { transition-duration: 80ms; } .nn-dialog-overlay .nn-dialog-surface { transform: none !important; } }
@media (prefers-reduced-transparency: reduce) { .nn-dialog-overlay { backdrop-filter: none; background: rgba(2,6,12,.92); } }
@media (prefers-contrast: more) { .nn-dialog-surface { background: #0b1118; border-color: #a6b3c0; } }
`;

function focusableElements(surface: HTMLElement): HTMLElement[] {
  return Array.from(surface.querySelectorAll<HTMLElement>(
    'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], summary, [tabindex]',
  )).filter(element => element.tabIndex >= 0 && !element.hidden && !element.closest('[inert], [aria-hidden="true"]')
    && window.getComputedStyle(element).display !== 'none' && window.getComputedStyle(element).visibility !== 'hidden');
}

export const AccessibleDialog: React.FC<AccessibleDialogProps> = ({
  isOpen, onClose, labelledBy, describedBy, initialFocusRef, variant = 'modal', className = '', children,
}) => {
  const [present, setPresent] = useState(isOpen);
  const [visible, setVisible] = useState(false);
  const overlayRef = useRef<HTMLDivElement>(null);
  const surfaceRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useLayoutEffect(() => {
    if (isOpen) {
      setPresent(true);
      const frame = requestAnimationFrame(() => setVisible(true));
      return () => cancelAnimationFrame(frame);
    }
    setVisible(false);
    const timer = window.setTimeout(() => setPresent(false), 180);
    return () => window.clearTimeout(timer);
  }, [isOpen]);

  useLayoutEffect(() => {
    if (overlayRef.current) overlayRef.current.inert = !isOpen;
  }, [isOpen, present]);

  useLayoutEffect(() => {
    const surface = surfaceRef.current;
    if (!isOpen || !present || !surface) return;

    const dialogId = Symbol('dialog');
    const trigger = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const firstDialog = openDialogs.length === 0;
    openDialogs.push(dialogId);

    const focusInside = () => {
      const requested = initialFocusRef?.current;
      const target = requested && surface.contains(requested) ? requested : focusableElements(surface)[0] || surface;
      target.focus({ preventScroll: true });
    };
    focusInside();

    if (firstDialog) {
      savedBodyOverflow = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      Array.from(document.body.children).forEach(element => {
        if (!(element instanceof HTMLElement) || element.hasAttribute('data-nn-dialog-root') || ['SCRIPT', 'STYLE'].includes(element.tagName)) return;
        backgroundState.set(element, { inert: element.inert, ariaHidden: element.getAttribute('aria-hidden') });
        element.inert = true;
        element.setAttribute('aria-hidden', 'true');
      });
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (openDialogs[openDialogs.length - 1] !== dialogId) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        closeRef.current();
      } else if (event.key === 'Tab') {
        const targets = focusableElements(surface);
        const first = targets[0];
        const last = targets[targets.length - 1];
        if (!first || !last) {
          event.preventDefault();
          surface.focus();
        } else if (event.shiftKey && (document.activeElement === first || document.activeElement === surface)) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && (document.activeElement === last || !surface.contains(document.activeElement))) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    const onFocus = (event: FocusEvent) => {
      if (openDialogs[openDialogs.length - 1] === dialogId && !surface.contains(event.target as Node)) focusInside();
    };
    document.addEventListener('keydown', onKeyDown, true);
    document.addEventListener('focusin', onFocus);

    return () => {
      document.removeEventListener('keydown', onKeyDown, true);
      document.removeEventListener('focusin', onFocus);
      const index = openDialogs.indexOf(dialogId);
      if (index >= 0) openDialogs.splice(index, 1);
      if (openDialogs.length === 0) {
        document.body.style.overflow = savedBodyOverflow;
        backgroundState.forEach((state, element) => {
          element.inert = state.inert;
          if (state.ariaHidden === null) element.removeAttribute('aria-hidden');
          else element.setAttribute('aria-hidden', state.ariaHidden);
        });
        backgroundState.clear();
      }
      if (trigger?.isConnected) {
        trigger.focus({ preventScroll: true });
        const remainingDialog = openDialogs[openDialogs.length - 1];
        // React can restore focus to the closing portal later in this commit.
        requestAnimationFrame(() => {
          const focused = document.activeElement;
          const focusLeftInPortal = focused === document.body || (focused !== null && surface.contains(focused));
          if (trigger.isConnected && !trigger.closest('[inert], [aria-hidden="true"]')
            && openDialogs[openDialogs.length - 1] === remainingDialog && focusLeftInPortal) {
            trigger.focus({ preventScroll: true });
          }
        });
      }
    };
  }, [isOpen, present, initialFocusRef]);

  if (!present) return null;

  return createPortal(
    <div
      ref={overlayRef}
      data-nn-dialog-root=""
      data-open={visible && isOpen}
      data-variant={variant}
      className="nn-dialog-overlay"
      aria-hidden={!isOpen ? true : undefined}
      style={!isOpen ? { pointerEvents: 'none' } : undefined}
      onClick={event => { if (isOpen && event.target === event.currentTarget) closeRef.current(); }}
    >
      <style>{dialogStyles}</style>
      <div ref={surfaceRef} role="dialog" aria-modal="true" aria-labelledby={labelledBy} aria-describedby={describedBy} tabIndex={-1} className={`nn-dialog-surface ${className}`}>
        {children}
      </div>
    </div>,
    document.body,
  );
};
