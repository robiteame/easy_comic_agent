import type React from 'react'
import { useRef } from 'react'
import gsap from 'gsap'
import { useGSAP } from '@gsap/react'

gsap.registerPlugin(useGSAP)

const frames = [
  { left: '11%', top: '14%', delay: 0 },
  { left: '17%', top: '19%', delay: 0.1 },
  { left: '83%', top: '13%', delay: 0.2 },
  { left: '88%', top: '20%', delay: 0.3 },
  { left: '78%', top: '78%', delay: 0.4 },
]

const GlobalPlayfulMotion: React.FC = () => {
  const scopeRef = useRef<HTMLDivElement | null>(null)

  useGSAP(
    () => {
      const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

      if (reduceMotion) {
        gsap.set('.global-motion-frame', { opacity: 0 })
        gsap.set('.global-motion-thread', { opacity: 0 })
        return
      }

      gsap.set('.global-motion-frame', {
        opacity: 0,
        rotate: -4,
        scale: 0.92,
        transformOrigin: '50% 50%',
      })

      gsap
        .timeline({ repeat: -1, repeatDelay: 1.4 })
        .to('.global-motion-frame', {
          opacity: 0.72,
          y: -8,
          rotate: 0,
          scale: 1,
          duration: 0.72,
          ease: 'power3.out',
          stagger: 0.11,
        })
        .to(
          '.global-motion-frame',
          {
            opacity: 0,
            y: -20,
            rotate: 4,
            duration: 0.64,
            ease: 'power2.in',
            stagger: 0.08,
          },
          '+=0.2',
        )

      gsap.to('.global-motion-thread path', {
        strokeDashoffset: -72,
        duration: 3.8,
        ease: 'sine.inOut',
        repeat: -1,
      })
    },
    { scope: scopeRef },
  )

  return (
    <div className="global-playful-motion" ref={scopeRef} aria-hidden="true">
      <svg className="global-motion-thread" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path d="M4 88 C24 70, 34 82, 50 62 S73 37, 96 23" />
      </svg>
      {frames.map((frame, _index) => (
        <span
          key={`${frame.left}-${frame.top}`}
          className="global-motion-frame"
          style={{
            left: frame.left,
            top: frame.top,
            animationDelay: `${frame.delay}s`,
          }}
        >
          <i />
          <i />
        </span>
      ))}
    </div>
  )
}

export default GlobalPlayfulMotion
