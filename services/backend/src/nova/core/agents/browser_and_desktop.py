"""NOVA Layer 5 Specialized Browser & Desktop Automation Controllers (Phase 13 / Phase 14 / ADR-008 / ADR-010 / FR-09 / FR-10 / Phase 15).

Plain-English Purpose (The Web-Browsing Specialist & Desktop Remote-Control Specialist):
In `NOVA_ADR.md` Phases 13 & 14 (pp. 44-46), the `Browser` and `Desktop` agents have special
safety and reliability rules because they interact with live websites and your computer screen:

1. **The Web-Browsing Specialist (`PlaywrightBrowserAutomationService` — Phase 13, pp. 44-45)**:
   - **Separate "Guest" Browser Profile by Default**: By default, NOVA uses an isolated browser
     profile (`profile_mode="isolated_automation"`) so websites can't touch your personal saved
     cookies or bank logins unless you explicitly opt in (`allow_authenticated_user_profile=True`).
   - **DOM-First, Vision-Second**: Reads clean HTML elements first (`playwright.extract`), and
     only falls back to `LocalVisionRouter` screen-grounding when a website has broken HTML.
   - **Strict CAPTCHA Rule (Non-Bypass Policy, p. 45)**: NOVA **never** tries to hack or bypass a
     CAPTCHA. If a webpage shows a CAPTCHA (`is_captcha_present`), NOVA immediately pauses and
     emits a `ClarificationProposal` (`kind="ask_user"`) asking you to solve the CAPTCHA manually!
   - **Untrusted Webpage Wrapping (`Phase 15 Layer 1`)**: Every scraped webpage is sanitized and
     wrapped in `<untrusted_data source_type="web_page">` so prompt injections on a website can
     never trick NOVA into thinking they are system commands.

2. **The Desktop Remote-Control Specialist (`CrossPlatformDesktopAutomationService` — Phase 14, pp. 45-46)**:
   - **Accessibility-First, Coordinate-Fallback Flagged**: Prefers official OS Accessibility APIs
     (`Windows UI Automation`, `macOS AX`, `Linux AT-SPI`) because they know *what* button they are
     clicking. If an old app has no accessibility tree, it falls back to pixel coordinates
     (`pyautogui.fallback_action`) and clearly flags `coordinate_fallback_engaged=True` to you.
   - **Hard Deny-List for Password Managers & Banking Apps (`Phase 14 Final Recommendation`, p. 46)**:
     Blocks the Desktop Agent from touching password managers (`1Password`, `Bitwarden`, `KeePass`,
     `Windows Credential Manager`) or banking apps by default (`SensitiveDesktopAppDeniedError`),
     unless you provide a separate, explicit per-window override!
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    PYDANTIC_V2 = False

from nova.core.domain.context import UntrustedDataItem, sanitize_untrusted_text
from nova.core.domain.events import EventType, NovaEvent
from nova.core.domain.primitives import (
    AgentRole,
    DataClassification,
    SensitivityTier,
)
from nova.core.domain.proposals import (
    ClarificationProposal,
    ToolCallProposal,
)
from nova.core.model_serving.multimodal import BoundingBox2D, LocalVisionRouter
from nova.core.platform.event_bus import EventPublisherPort


class BrowserSecurityError(PermissionError):
    """Raised when a Browser Agent operation violates Phase 13 profile isolation or security rules."""


class SensitiveDesktopAppDeniedError(PermissionError):
    """Raised when the Desktop Agent attempts to automate a password manager or banking app (Phase 14, p. 46)."""


class BrowserProfileMode(str, Enum):
    """Browser context isolation modes defined in NOVA_ADR.md Phase 13 (p. 45)."""

    ISOLATED_AUTOMATION = "isolated_automation"          # Default: separate cookies/storage
    AUTHENTICATED_USER_OPT_IN = "authenticated_user_opt_in"  # Requires explicit per-task user opt-in


class DesktopPlatformKind(str, Enum):
    """Cross-platform OS accessibility backends defined in NOVA_ADR.md Section 12.1 & Phase 14 (p. 45)."""

    WINDOWS_UIA = "windows_uia"  # Windows UI Automation
    MACOS_AX = "macos_ax"        # macOS Accessibility API
    LINUX_ATSPI = "linux_atspi"  # Linux AT-SPI


class WebPageSnapshot(BaseModel):
    """Structured representation of a loaded browser page before untrusted data wrapping (Phase 13)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    url: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    dom_text: str = Field(default="")
    has_semantic_dom: bool = Field(
        default=True,
        description="False when page is a custom canvas/WebGL UI requiring Phase 12 Vision fallback",
    )
    captcha_detected: bool = Field(
        default=False,
        description="Phase 13 (p. 45): True if a CAPTCHA challenge is present on the page",
    )
    captcha_provider: Optional[str] = Field(default=None)


class PlaywrightBrowserAutomationService:
    """Phase 13 Browser Automation Controller enforcing profile isolation, CAPTCHA escalation, and DOM/Vision fallback."""

    def __init__(
        self,
        vision_router: Optional[LocalVisionRouter] = None,
        event_bus: Optional[EventPublisherPort] = None,
        allow_authenticated_user_profile: bool = False,
    ) -> None:
        self._vision = vision_router or LocalVisionRouter()
        self._event_bus = event_bus
        self._allow_auth_profile = allow_authenticated_user_profile

    @property
    def allow_authenticated_user_profile(self) -> bool:
        return self._allow_auth_profile

    def verify_profile_access(self, requested_mode: BrowserProfileMode) -> BrowserProfileMode:
        """Enforce Phase 13 Final Recommendation (p. 45): personal profile requires explicit opt-in."""
        if (
            requested_mode == BrowserProfileMode.AUTHENTICATED_USER_OPT_IN
            and not self._allow_auth_profile
        ):
            raise BrowserSecurityError(
                "Phase 13 / ADR-010 violation: Browser Agent cannot use the user's personal "
                "authenticated browser profile unless allow_authenticated_user_profile=True is "
                "explicitly opted in for this task."
            )
        return requested_mode

    def inspect_page_or_escalate_captcha(
        self,
        *,
        task_id: str,
        session_id: str,
        node_id: str,
        snapshot: WebPageSnapshot,
    ) -> UntrustedDataItem | ClarificationProposal:
        """Process a loaded page: escalate CAPTCHAs to the user, or return a sanitized `<untrusted_data>` item."""
        # 1. Phase 13 Non-Goal / Policy (p. 45): Never bypass CAPTCHAs; surface to user!
        if snapshot.captcha_detected:
            provider = snapshot.captcha_provider or "CAPTCHA"
            return ClarificationProposal(
                agent_role=AgentRole.BROWSER,
                node_id=node_id,
                thought=(
                    f"Detected {provider} challenge on {snapshot.url}. Per Phase 13 policy, "
                    "NOVA never attempts automated CAPTCHA bypass and must ask the user."
                ),
                question=(
                    f"The page '{snapshot.title}' ({snapshot.url}) presented a {provider} challenge. "
                    "Please complete the CAPTCHA in the browser window so I can continue."
                ),
                missing_fields=("manual_captcha_completion",),
                options=("I have completed the CAPTCHA", "Cancel this web task"),
            )

        # 2. Wrap scraped page content strictly as passive `<untrusted_data source_type="web_page">` (Phase 15 Layer 1)
        sanitized_dom = sanitize_untrusted_text(
            snapshot.dom_text or f"[Empty DOM on {snapshot.url}]"
        )
        return UntrustedDataItem(
            item_id=f"web_{node_id}",
            source_type="web_page",
            source_uri=snapshot.url,
            data_classification=DataClassification.PUBLIC,
            content=f"Page Title: {sanitize_untrusted_text(snapshot.title)}\n{sanitized_dom}",
        )

    def build_click_proposal_with_vision_fallback(
        self,
        *,
        node_id: str,
        snapshot: WebPageSnapshot,
        target_selector: Optional[str],
        target_visual_description: str,
    ) -> Tuple[ToolCallProposal, bool]:
        """Build a DOM `playwright.click` proposal, or fall back to Vision grounding if DOM markup is absent.

        Returns `(proposal, used_vision_fallback)`.
        """
        if snapshot.has_semantic_dom and target_selector:
            return (
                ToolCallProposal(
                    agent_role=AgentRole.BROWSER,
                    node_id=node_id,
                    thought=f"Click semantic DOM element '{target_selector}' on {snapshot.url}.",
                    tool_name="playwright.click",
                    arguments={"selector": target_selector},
                    declared_sensitivity=SensitivityTier.REVERSIBLE_WRITE,
                    rationale=f"Click '{target_visual_description}' via DOM selector {target_selector}",
                ),
                False,
            )

        # Fallback: Page has poor semantic markup -> use Phase 12 Vision grounding!
        grounding = self._vision.ground_ui_element(target_visual_description)
        cx, cy = grounding.bounding_box.center_point
        synthetic_selector = f"vision_coords({cx},{cy})"
        return (
            ToolCallProposal(
                agent_role=AgentRole.BROWSER,
                node_id=node_id,
                thought=(
                    f"Page {snapshot.url} lacks semantic DOM markup; used {grounding.engine.value} "
                    f"vision grounding to locate '{target_visual_description}' at ({cx}, {cy})."
                ),
                tool_name="playwright.click",
                arguments={"selector": synthetic_selector},
                declared_sensitivity=SensitivityTier.REVERSIBLE_WRITE,
                rationale=(
                    f"Vision-fallback click on '{target_visual_description}' at ({cx}, {cy})"
                ),
            ),
            True,
        )


# Default deny-list of sensitive application categories (Phase 14 Final Recommendation, p. 46)
DEFAULT_SENSITIVE_APP_DENY_KEYWORDS: Tuple[str, ...] = (
    "1password",
    "bitwarden",
    "keepass",
    "lastpass",
    "credential manager",
    "keychain access",
    "bank",
    "banking",
    "chase",
    "netbanking",
    "crypto wallet",
    "ledger live",
)


class DesktopElementTarget(BaseModel):
    """Represents a target desktop UI control via OS Accessibility Tree or coordinate fallback (Phase 14)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    window_title: str = Field(..., min_length=1)
    accessibility_element_id: Optional[str] = Field(
        default=None,
        description="Structured OS accessibility element ID (primary path when present)",
    )
    visual_description: str = Field(
        ...,
        min_length=1,
        description="Natural-language control description used if coordinate fallback is required",
    )


class DesktopAutomationPlanResult(BaseModel):
    """Result of planning a desktop automation action across Accessibility vs. PyAutoGUI fallback (Phase 14)."""

    if PYDANTIC_V2:
        model_config = ConfigDict(frozen=True, extra="forbid")
    else:

        class Config:
            allow_mutation = False
            extra = "forbid"

    proposal: ToolCallProposal = Field(...)
    platform_backend: DesktopPlatformKind = Field(...)
    coordinate_fallback_engaged: bool = Field(
        ...,
        description="Phase 14 (p. 46): True when PyAutoGUI coordinate fallback was used instead of Accessibility APIs",
    )
    user_warning_banner: Optional[str] = Field(
        default=None,
        description="Warning shown to the user when lower-reliability coordinate fallback is engaged",
    )


class CrossPlatformDesktopAutomationService:
    """Phase 14 Desktop Automation Controller enforcing Accessibility-First, PyAutoGUI warning flags, and App Deny-Lists."""

    def __init__(
        self,
        platform_backend: DesktopPlatformKind = DesktopPlatformKind.WINDOWS_UIA,
        vision_router: Optional[LocalVisionRouter] = None,
        event_bus: Optional[EventPublisherPort] = None,
        denied_app_keywords: Sequence[str] = DEFAULT_SENSITIVE_APP_DENY_KEYWORDS,
        explicitly_overridden_windows: Sequence[str] = (),
    ) -> None:
        self._platform = platform_backend
        self._vision = vision_router or LocalVisionRouter()
        self._event_bus = event_bus
        self._denied_keywords: Tuple[str, ...] = tuple(
            k.strip().lower() for k in denied_app_keywords if k.strip()
        )
        self._overridden_windows: FrozenSet[str] = frozenset(
            w.strip().lower() for w in explicitly_overridden_windows if w.strip()
        )

    @property
    def platform_backend(self) -> DesktopPlatformKind:
        return self._platform

    def verify_window_permitted(
        self,
        window_title: str,
        *,
        task_id: str = "task_desktop",
        session_id: str = "sess_desktop",
        node_id: str = "node_desktop",
    ) -> None:
        """Block automation on password managers & banking apps unless separately overridden (Phase 14, p. 46)."""
        normalized = window_title.strip().lower()
        if normalized in self._overridden_windows:
            return

        for keyword in self._denied_keywords:
            if keyword in normalized:
                if self._event_bus is not None:
                    self._event_bus.publish(
                        NovaEvent(
                            event_type=EventType.SECURITY_VIOLATION,
                            task_id=task_id,
                            session_id=session_id,
                            node_id=node_id,
                            agent_role=AgentRole.DESKTOP,
                            payload={
                                "reason": "sensitive_desktop_application_blocked",
                                "window_title": window_title,
                                "matched_deny_keyword": keyword,
                            },
                        )
                    )
                raise SensitiveDesktopAppDeniedError(
                    f"Phase 14 / ADR-010 security deny-list: Desktop automation on window "
                    f"{window_title!r} (matched sensitive category {keyword!r}) is blocked by "
                    "default. Requires a separate, explicit per-application user override."
                )

    def build_desktop_action_proposal(
        self,
        *,
        node_id: str,
        target: DesktopElementTarget,
        task_id: str = "task_desktop",
        session_id: str = "sess_desktop",
    ) -> DesktopAutomationPlanResult:
        """Create an Accessibility-API tool proposal (primary) or flagged PyAutoGUI fallback proposal (Phase 14)."""
        # 1. Always enforce the Sensitive Application Deny-List first!
        self.verify_window_permitted(
            target.window_title,
            task_id=task_id,
            session_id=session_id,
            node_id=node_id,
        )

        # 2. Primary Path: OS Accessibility API (`accessibility.invoke_element`)
        if target.accessibility_element_id:
            proposal = ToolCallProposal(
                agent_role=AgentRole.DESKTOP,
                node_id=node_id,
                thought=(
                    f"Invoke accessibility element {target.accessibility_element_id!r} in window "
                    f"{target.window_title!r} via {self._platform.value}."
                ),
                tool_name="accessibility.invoke_element",
                arguments={
                    "window_title": target.window_title,
                    "element_id": target.accessibility_element_id,
                },
                declared_sensitivity=SensitivityTier.REVERSIBLE_WRITE,
                rationale=(
                    f"Invoke '{target.visual_description}' via OS accessibility API ({self._platform.value})"
                ),
            )
            return DesktopAutomationPlanResult(
                proposal=proposal,
                platform_backend=self._platform,
                coordinate_fallback_engaged=False,
                user_warning_banner=None,
            )

        # 3. Fallback Path: App has no accessibility element -> use Vision grounding + PyAutoGUI coordinate click!
        # Per Phase 14 Final Recommendation (p. 46), clearly flag this lower-reliability mode to the user
        # and escalate sensitivity to IRREVERSIBLE_WRITE so the Permission Gate asks for human confirmation.
        grounding = self._vision.ground_ui_element(target.visual_description)
        cx, cy = grounding.bounding_box.center_point
        warning = (
            f"LOW-RELIABILITY COORDINATE FALLBACK ENGAGED: Window {target.window_title!r} "
            f"did not expose an accessibility tree for {target.visual_description!r}. "
            f"Falling back to PyAutoGUI click at screen coordinates ({cx}, {cy})."
        )
        proposal = ToolCallProposal(
            agent_role=AgentRole.DESKTOP,
            node_id=node_id,
            thought=warning,
            tool_name="pyautogui.fallback_action",
            arguments={
                "action": "click",
                "x": cx,
                "y": cy,
            },
            declared_sensitivity=SensitivityTier.IRREVERSIBLE_WRITE,
            rationale=warning,
        )
        return DesktopAutomationPlanResult(
            proposal=proposal,
            platform_backend=self._platform,
            coordinate_fallback_engaged=True,
            user_warning_banner=warning,
        )


__all__ = [
    "BrowserProfileMode",
    "BrowserSecurityError",
    "CrossPlatformDesktopAutomationService",
    "DEFAULT_SENSITIVE_APP_DENY_KEYWORDS",
    "DesktopAutomationPlanResult",
    "DesktopElementTarget",
    "DesktopPlatformKind",
    "PlaywrightBrowserAutomationService",
    "SensitiveDesktopAppDeniedError",
    "WebPageSnapshot",
]