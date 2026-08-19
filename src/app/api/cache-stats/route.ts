import { NextResponse } from 'next/server';
import { getCacheStats, discardWeakResponses } from '@/lib/cloud-cache';

export async function GET() {
  try {
    const stats = await getCacheStats();
    return NextResponse.json({
      enabled: stats !== null,
      ...stats,
    });
  } catch {
    return NextResponse.json({ enabled: false });
  }
}

export async function POST() {
  try {
    const discarded = await discardWeakResponses();
    return NextResponse.json({ discarded });
  } catch {
    return NextResponse.json({ discarded: 0 }, { status: 500 });
  }
}
