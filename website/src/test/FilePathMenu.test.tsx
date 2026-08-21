import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { screen, fireEvent, waitFor } from '@testing-library/react'
import { renderWithProviders } from './helpers'
import FilePathMenu from '../components/FilePathMenu'

// ── Mocks ────────────────────────────────────────────────────────────────────

const brandingEnv = vi.hoisted(() => ({ directLocal: true }))
const platformEnv = vi.hoisted(() => ({ value: 'other' as 'other' | 'darwin' | 'windows' }))

vi.mock('../hooks/useBranding', () => ({
  useBranding: () => ({ botName: 'Test', avatar: '', directLocal: brandingEnv.directLocal }),
}))

// The reveal label is platform-aware (names Finder / File Explorer on the
// gateway's own OS). Drive it explicitly so the label assertions are stable
// regardless of the test host.
vi.mock('../hooks/useGatewayPlatform', () => ({
  useGatewayPlatform: () => platformEnv.value,
}))

vi.mock('../api/client', () => ({
  api: {
    revealPath: vi.fn().mockResolvedValue({ ok: true }),
  },
}))

vi.mock('../utils/clipboard', () => ({
  copyToClipboard: vi.fn(),
}))

import { api } from '../api/client'
import { copyToClipboard } from '../utils/clipboard'

// ── Helpers ──────────────────────────────────────────────────────────────────

beforeEach(() => {
  vi.clearAllMocks()
  brandingEnv.directLocal = true
  platformEnv.value = 'other'
})

afterEach(() => {
  brandingEnv.directLocal = true
  platformEnv.value = 'other'
})

function rightClick(el: Element) {
  fireEvent.contextMenu(el)
}

// ── FilePathMenu (right-click wrapper) ───────────────────────────────────────

describe('FilePathMenu', () => {
  const TEST_PATH = '/home/user/project/report.md'

  describe('when directLocal is true', () => {
    it('renders all three items: open, reveal, copy path', async () => {
      renderWithProviders(
        <FilePathMenu filePath={TEST_PATH}>
          <span data-testid="trigger">report.md</span>
        </FilePathMenu>,
      )

      rightClick(screen.getByTestId('trigger'))

      await waitFor(() => {
        expect(screen.getByText('Open with default app')).toBeInTheDocument()
      })
      expect(screen.getByText('Show in file manager')).toBeInTheDocument()
      expect(screen.getByText('Copy path')).toBeInTheDocument()
    })

    it('calls revealPath with "open" when Open item is selected', async () => {
      renderWithProviders(
        <FilePathMenu filePath={TEST_PATH}>
          <span data-testid="trigger">report.md</span>
        </FilePathMenu>,
      )

      rightClick(screen.getByTestId('trigger'))

      await waitFor(() => {
        expect(screen.getByText('Open with default app')).toBeInTheDocument()
      })
      fireEvent.click(screen.getByText('Open with default app'))

      await waitFor(() => {
        expect(api.revealPath).toHaveBeenCalledWith(TEST_PATH, 'open')
      })
    })

    it('calls revealPath with "reveal" when Reveal item is selected', async () => {
      renderWithProviders(
        <FilePathMenu filePath={TEST_PATH}>
          <span data-testid="trigger">report.md</span>
        </FilePathMenu>,
      )

      rightClick(screen.getByTestId('trigger'))

      await waitFor(() => {
        expect(screen.getByText('Show in file manager')).toBeInTheDocument()
      })
      fireEvent.click(screen.getByText('Show in file manager'))

      await waitFor(() => {
        expect(api.revealPath).toHaveBeenCalledWith(TEST_PATH, 'reveal')
      })
    })

    it('calls copyToClipboard when Copy path item is selected', async () => {
      renderWithProviders(
        <FilePathMenu filePath={TEST_PATH}>
          <span data-testid="trigger">report.md</span>
        </FilePathMenu>,
      )

      rightClick(screen.getByTestId('trigger'))

      await waitFor(() => {
        expect(screen.getByText('Copy path')).toBeInTheDocument()
      })
      fireEvent.click(screen.getByText('Copy path'))

      expect(copyToClipboard).toHaveBeenCalledWith(TEST_PATH)
    })
  })

  describe('when directLocal is false (remote session)', () => {
    beforeEach(() => { brandingEnv.directLocal = false })

    it('hides open and reveal items, shows only copy path', async () => {
      renderWithProviders(
        <FilePathMenu filePath={TEST_PATH}>
          <span data-testid="trigger">report.md</span>
        </FilePathMenu>,
      )

      rightClick(screen.getByTestId('trigger'))

      await waitFor(() => {
        expect(screen.getByText('Copy path')).toBeInTheDocument()
      })
      expect(screen.queryByText('Open with default app')).not.toBeInTheDocument()
      expect(screen.queryByText('Show in file manager')).not.toBeInTheDocument()
    })
  })

  describe('platform-aware reveal label', () => {
    it('names Finder on a macOS gateway', async () => {
      platformEnv.value = 'darwin'
      renderWithProviders(
        <FilePathMenu filePath={TEST_PATH}>
          <span data-testid="trigger">report.md</span>
        </FilePathMenu>,
      )

      rightClick(screen.getByTestId('trigger'))

      await waitFor(() => {
        expect(screen.getByText('Open in Finder')).toBeInTheDocument()
      })
      // The reveal label follows the gateway OS; the generic wording is gone.
      expect(screen.queryByText('Show in file manager')).not.toBeInTheDocument()
      fireEvent.click(screen.getByText('Open in Finder'))
      await waitFor(() => {
        expect(api.revealPath).toHaveBeenCalledWith(TEST_PATH, 'reveal')
      })
    })

    it('names File Explorer on a Windows gateway', async () => {
      platformEnv.value = 'windows'
      renderWithProviders(
        <FilePathMenu filePath={TEST_PATH}>
          <span data-testid="trigger">report.md</span>
        </FilePathMenu>,
      )

      rightClick(screen.getByTestId('trigger'))

      await waitFor(() => {
        expect(screen.getByText('Open in File Explorer')).toBeInTheDocument()
      })
    })
  })

  describe('Windows suppresses "Open with default app"', () => {
    // The gateway's files.py refuses the launch-by-association verb on Windows
    // and degrades an `open` to a clipboard copy, so the row must not appear
    // there — it would promise a launch the backend never performs. Reveal
    // (which does work) and Copy path stay.
    it('hides Open on a Windows gateway but keeps reveal + copy', async () => {
      platformEnv.value = 'windows'
      renderWithProviders(
        <FilePathMenu filePath={TEST_PATH}>
          <span data-testid="trigger">report.md</span>
        </FilePathMenu>,
      )

      rightClick(screen.getByTestId('trigger'))

      await waitFor(() => {
        expect(screen.getByText('Open in File Explorer')).toBeInTheDocument()
      })
      expect(screen.queryByText('Open with default app')).not.toBeInTheDocument()
      expect(screen.getByText('Copy path')).toBeInTheDocument()
    })

    it('shows Open on a macOS gateway', async () => {
      platformEnv.value = 'darwin'
      renderWithProviders(
        <FilePathMenu filePath={TEST_PATH}>
          <span data-testid="trigger">report.md</span>
        </FilePathMenu>,
      )

      rightClick(screen.getByTestId('trigger'))

      await waitFor(() => {
        expect(screen.getByText('Open with default app')).toBeInTheDocument()
      })
    })
  })

  describe('directory paths', () => {
    it('hides "Open with default app" for a directory but keeps reveal + copy', async () => {
      renderWithProviders(
        <FilePathMenu filePath="/home/user/project" kind="dir">
          <span data-testid="trigger">project</span>
        </FilePathMenu>,
      )

      rightClick(screen.getByTestId('trigger'))

      await waitFor(() => {
        expect(screen.getByText('Show in file manager')).toBeInTheDocument()
      })
      expect(screen.getByText('Copy path')).toBeInTheDocument()
      // A directory cannot be "opened" — /api/reveal 400s an open on a dir.
      expect(screen.queryByText('Open with default app')).not.toBeInTheDocument()
    })
  })
})

// ── Item rows (rendered via the FilePathMenu wrapper) ────────────────────────

describe('FilePathMenu item rows', () => {
  const TEST_PATH = '/tmp/demo.html'

  // Exercise the item rows through the public FilePathMenu wrapper, which
  // renders them verbatim. FilePathMenuItems is a private building block (no
  // export), so the wrapper is its only entry point. Gating is driven by the
  // branding mock (brandingEnv.directLocal) — the items read useBranding().
  function renderContext(props: { directLocal?: boolean; kind?: 'file' | 'dir' }) {
    if (props.directLocal !== undefined) brandingEnv.directLocal = props.directLocal
    return renderWithProviders(
      <FilePathMenu filePath={TEST_PATH} kind={props.kind}>
        <span data-testid="ctx-trigger">file.txt</span>
      </FilePathMenu>,
    )
  }

  it('renders items when directLocal', async () => {
    renderContext({ directLocal: true })

    rightClick(screen.getByTestId('ctx-trigger'))

    await waitFor(() => {
      expect(screen.getByText('Open with default app')).toBeInTheDocument()
    })
    expect(screen.getByText('Show in file manager')).toBeInTheDocument()
    expect(screen.getByText('Copy path')).toBeInTheDocument()
  })

  it('hides open/reveal when remote', async () => {
    renderContext({ directLocal: false })

    rightClick(screen.getByTestId('ctx-trigger'))

    await waitFor(() => {
      expect(screen.getByText('Copy path')).toBeInTheDocument()
    })
    expect(screen.queryByText('Open with default app')).not.toBeInTheDocument()
    expect(screen.queryByText('Show in file manager')).not.toBeInTheDocument()
  })

  it('calls revealPath("open") on open item click', async () => {
    renderContext({ directLocal: true })

    rightClick(screen.getByTestId('ctx-trigger'))

    await waitFor(() => {
      expect(screen.getByText('Open with default app')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Open with default app'))

    await waitFor(() => {
      expect(api.revealPath).toHaveBeenCalledWith(TEST_PATH, 'open')
    })
  })

  it('calls copyToClipboard on copy path click', async () => {
    renderContext({ directLocal: true })

    rightClick(screen.getByTestId('ctx-trigger'))

    await waitFor(() => {
      expect(screen.getByText('Copy path')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Copy path'))

    expect(copyToClipboard).toHaveBeenCalledWith(TEST_PATH)
  })

  describe('aria labels', () => {
    it('each item has an accessible aria-label', async () => {
      renderContext({ directLocal: true })

      rightClick(screen.getByTestId('ctx-trigger'))

      await waitFor(() => {
        expect(screen.getByLabelText('Open with default app')).toBeInTheDocument()
      })
      expect(screen.getByLabelText('Show in file manager')).toBeInTheDocument()
      expect(screen.getByLabelText('Copy path')).toBeInTheDocument()
    })
  })
})
