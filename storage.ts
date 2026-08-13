export interface SavedConversation {
  id: string;
  title: string;
  messages: { role: 'user' | 'assistant'; content: string }[];
  model: string;
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = 'nexusai_conversations';
const ACTIVE_KEY = 'nexusai_active_conversation';

export function loadConversations(): SavedConversation[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveConversation(conversation: SavedConversation): void {
  const conversations = loadConversations();
  const idx = conversations.findIndex(c => c.id === conversation.id);
  if (idx >= 0) {
    conversations[idx] = conversation;
  } else {
    conversations.unshift(conversation);
  }
  // Keep max 50 conversations
  const trimmed = conversations.slice(0, 50);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
}

export function deleteConversation(id: string): void {
  const conversations = loadConversations().filter(c => c.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  if (getActiveConversationId() === id) {
    localStorage.removeItem(ACTIVE_KEY);
  }
}

export function getActiveConversationId(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(ACTIVE_KEY);
}

export function setActiveConversationId(id: string | null): void {
  if (typeof window === 'undefined') return;
  if (id) {
    localStorage.setItem(ACTIVE_KEY, id);
  } else {
    localStorage.removeItem(ACTIVE_KEY);
  }
}

export function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export function generateTitle(firstMessage: string): string {
  const cleaned = firstMessage.replace(/\n/g, ' ').trim();
  return cleaned.length > 40 ? cleaned.slice(0, 40) + '...' : cleaned;
}
