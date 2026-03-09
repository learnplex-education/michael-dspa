import NextAuth from "next-auth";
import Google from "next-auth/providers/google";

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    Google({
      // This forces Google to provide the id_token if it's being shy
      authorization: {
        params: {
          scope: "openid email profile",
        },
      },
    }),
  ],
  callbacks: {
    /**
     * Berkeley-only gate: only allow @berkeley.edu emails to sign in.
     */
    async signIn({ profile }) {
      const email = (profile as { email?: string } | null)?.email ?? "";
      if (!email.endsWith("@berkeley.edu")) {
        return false;
      }
      return true;
    },

    /**
     * Token callback: capture Google's ID token on initial sign-in
     * so we can forward it to the FastAPI backend.
     */
    async jwt({ token, account }) {
      // account is only defined on initial sign-in
      if (account && (account as any).id_token) {
        // Persist the Google ID token on our JWT
        (token as any).idToken = (account as any).id_token;
      }
      return token;
    },

    /**
     * Session callback: expose the ID token to the client session
     * so the chat UI can send it as a Bearer token to FastAPI.
     */
    async session({ session, token }) {
      (session as any).idToken = (token as any).idToken;
      return session;
    },
  },
});

